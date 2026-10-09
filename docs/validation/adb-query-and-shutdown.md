# Bounded process queries and outstanding shutdown observations

During deployment, a task-owned ADB client remained waiting on a read of `/proc/28108/cmdline` after the server no longer existed. Another ADB shell query worked and the device was still listed as connected. The client was identified by its exact command and parent process, then only that stale query client was stopped; the ADB server was not restarted and weights/keys were retained. The original deployment command then continued rather than being restarted.

`serve_model.ps1` now uses `Invoke-ProjectAdbQuery` for the three ownership queries. They have a ten-second timeout and run hidden, with separate stdout/stderr capture. A timeout stops only that query client and throws an unknown-state error: it does not pretend the device server is absent. Transport/remote-command failures also fail the query. Device-side ownership checks still require the project working directory and exact model command.

Manual checks cover normal output, a forced 200 ms timeout on a two-second read-only sleep command, and server Status/Stop/Start. The forced timeout produces the expected unknown-state exception.

## Crash observations are not resolved by this host-side change

The device crash buffer contains older HwBinder-thread SIGSEGV records for server PIDs 7,648, 17,330, and 19,100, at device timestamps 04:32:40, 05:35:57, and 05:48:42 on October 10. There was no matching record for the stale-query PID 28,108. The buffer alone does not establish their causes or whether each occurred during shutdown or active inference.

After the packed-rounding API regressions, a controlled Stop/Start of owned server PID 30,645 completed and its crash-buffer query returned no matching record. That single clean attempt does not resolve the older crashes or prove sustained stability. Shutdown lifecycle and driver-unload behavior still require investigation; the relevant dynamic library wrapper currently uses `dlclose`, but no causal claim is established.

The API server was restarted after this check. Keep standalone NPU probes and model inference serialized during further validation, and never overwrite mapped runtime libraries.
