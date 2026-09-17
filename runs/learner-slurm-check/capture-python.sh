#!/bin/bash
printf "%s\0" "$@" > "$ATC_CAPTURE_ARGS"
