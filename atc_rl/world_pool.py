"""SB3 vector interface preserving world boundaries and individual aircraft terminals."""
from pathlib import Path
import multiprocessing as mp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from atc_rl.world_worker import worker,POPULATION

class WorldPool(VecEnv):
    def __init__(self,workers,directory,guidance=False,filter=False,progress_scale=0.0,action_reference="direct",conflict_features=False,mask_conflict_features=False,static_filter=False,traffic_position_scale=1.0):
        if action_reference not in ("direct","goal_offset"):raise ValueError("Unknown action reference")
        if mask_conflict_features and not conflict_features:raise ValueError("Masking requires conflict_features")
        from atc_rl.traffic_scaling import position_scale
        traffic_position_scale=position_scale({'traffic_position_scale':traffic_position_scale})
        self.action_reference=action_reference
        if workers<1:raise ValueError('At least one simulator worker is required')
        self.connections=[];self.processes=[];self.waiting=False;self.closed=False
        context=mp.get_context('spawn')
        try:
            for index in range(workers):
                parent,child=context.Pipe()
                process=context.Process(target=worker,args=(child,{'guidance':bool(guidance),'filter':bool(filter),'static_filter':bool(static_filter),'progress_scale':float(progress_scale),'action_reference':action_reference,'conflict_features':bool(conflict_features),'mask_conflict_features':bool(mask_conflict_features),'traffic_position_scale':traffic_position_scale},
                    str(Path(directory).resolve()/f'world-{index}')),daemon=True)
                process.start();child.close();self.connections.append(parent);self.processes.append(process)
            specifications=[self._receive(c,expected='ready') for c in self.connections]
            original,action,self.runtime=specifications[0]
            assert all(s[0]==original and s[1]==action and s[2]==self.runtime for s in specifications)
            actor_low=np.append(original.low,0).astype(np.float32)
            actor_high=np.append(original.high,1).astype(np.float32)
            critic_low=np.concatenate((np.tile(actor_low,POPULATION),np.zeros(2*POPULATION,dtype=np.float32)))
            critic_high=np.concatenate((np.tile(actor_high,POPULATION),np.ones(2*POPULATION,dtype=np.float32)))
            observation=spaces.Dict({'actor':spaces.Box(actor_low,actor_high,dtype=np.float32),
                                    'critic':spaces.Box(critic_low,critic_high,dtype=np.float32)})
            super().__init__(workers*POPULATION,observation,action)
        except BaseException:
            self.close();raise

    @staticmethod
    def _receive(connection,expected='ok'):
        status,payload=connection.recv()
        if status=='error':raise RuntimeError('Simulator worker failed:\n'+payload)
        if status!=expected:raise RuntimeError(f'Unexpected worker response: {status}')
        return payload

    @staticmethod
    def _combine(observations):
        return {key:np.concatenate([obs[key] for obs in observations],axis=0) for key in observations[0]}

    def reset(self):
        if self.waiting:raise RuntimeError('Cannot reset while a world step is pending')
        for index,c in enumerate(self.connections):c.send(('reset',self._seeds[index*POPULATION]))
        results=[self._receive(c) for c in self.connections]
        self.reset_infos=[info for _,infos in results for info in infos]
        self._reset_seeds();self._reset_options()
        return self._combine([obs for obs,_ in results])

    def step_async(self,actions):
        if self.waiting:raise RuntimeError('A vector step is already pending')
        actions=np.asarray(actions,dtype=np.float32)
        if actions.shape!=(self.num_envs,2) or not np.isfinite(actions).all() or (np.abs(actions)>1).any():
            raise ValueError('Expected finite bounded heading/speed commands for every aircraft slot')
        for index,c in enumerate(self.connections):c.send(('step',actions[index*POPULATION:(index+1)*POPULATION]))
        self.waiting=True

    def step_wait(self):
        results=[self._receive(c) for c in self.connections];self.waiting=False
        self.reset_infos=[info for result in results for info in result[4]]
        return (self._combine([r[0] for r in results]),np.concatenate([r[1] for r in results]),
                np.concatenate([r[2] for r in results]),[info for r in results for info in r[3]])

    def get_attr(self,attr_name,indices=None):
        chosen=list(self._get_indices(indices))
        worlds=sorted({i//POPULATION for i in chosen})
        for index in worlds:self.connections[index].send(('getattr',attr_name))
        values={index:self._receive(self.connections[index]) for index in worlds}
        return [values[index//POPULATION] for index in chosen]

    def actor_schema(self):
        """Read the actual simulator layout without resetting or advancing a world."""
        if self.waiting:
            raise RuntimeError("Wait for the pending world step before reading its schema")
        for connection in self.connections:
            connection.send(('observation_schema',None))
        schemas=[self._receive(connection) for connection in self.connections]
        if any(schema!=schemas[0] for schema in schemas[1:]):
            raise ValueError("World observation layouts differ")
        return schemas[0]

    def goal_actions(self,observations):
        """Execute the frozen classical benchmark for adapter/evaluation checks."""
        if self.action_reference!="direct":raise ValueError("The fixed benchmark requires direct action commands")
        for index,connection in enumerate(self.connections):
            connection.send(('goal_actions',observations['actor'][index*POPULATION:(index+1)*POPULATION,:-1]))
        return np.concatenate([self._receive(c) for c in self.connections])

    def set_attr(self,attr_name,value,indices=None):
        raise NotImplementedError('Simulator configuration is fixed for a training run')

    def env_method(self,method_name,*method_args,indices=None,**method_kwargs):
        raise NotImplementedError('Use the explicit vector reset and step operations')

    def env_is_wrapped(self,wrapper_class,indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        if self.closed:return
        self.closed=True
        for connection in self.connections:
            try:connection.send(('close',None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():process.terminate();process.join(timeout=3)
        for connection in self.connections:connection.close()
