# Python Backend Active Health Checks

This document describes the active health checking mechanism implemented in the Python Backend for Triton Inference Server. This feature ensures that the server accurately reflects the state of the backend process and the model logic, preventing traffic from being routed to crashed or unresponsive models.

## 1. Overview

The standard Triton health check (`/v2/models/<model>/ready`) historically relied on cached state managed by the `ModelRepositoryManager`. This was insufficient for the Python backend because:
1.  **Silent Crashes**: If the isolated Python stub process crashed (e.g., SEGFAULT, OOM), the parent Triton process would not immediately know, and the API would continue returning `200 OK`.
2.  **Custom Logic**: Users had no way to define their own "Readiness" logic (e.g., checking if a database connection is active) within their Python model code.

This update introduces an **Active Health Check** mechanism that probes the backend at runtime when the readiness endpoint is queried.

## 2. Architecture

The solution involves changes across Triton Core and the Python Backend.

### 2.1 Triton Core Integration
*   **New API**: `TRITONBACKEND_ModelInstanceReady` was added to the backend C API.
*   **Server Logic**: `InferenceServer::ModelIsReady` now calls this API (if implemented by the backend) in addition to checking the internal state.
*   **Aggregation**: If a model has multiple instances, the health status is aggregated. If *any* instance reports "Not Ready", the entire model is marked "Not Ready".

### 2.2 Python Backend Implementation
The Python backend implements `TRITONBACKEND_ModelInstanceReady` with a multi-stage check:

#### Stage 1: Process Existence (OS Level)
*   **Mechanism**: Calls `waitpid(stub_pid, WNOHANG)` on the backend stub process.
*   **Purpose**: Instantly detects if the process has crashed (SIGSEGV, SIGABRT) or been killed (SIGKILL, OOM Killer).
*   **Behavior**: If the process is found to be a zombie or missing, it is reaped, and the check returns `TRITONSERVER_ERROR_INTERNAL` ("Stub process is not healthy").

#### Stage 2: User-Defined Readiness (Application Level)
*   **Mechanism**: Sends an IPC (Inter-Process Communication) message `PYTHONSTUB_CheckIsModelReady` to the Python stub.
*   **User Interface**: Users can define an optional `is_model_ready(self) -> bool` method in their `TritonPythonModel` class.
*   **Behavior**:
    *   If the method exists, it is executed.
        *   Returns `True`: Model is Ready.
        *   Returns `False` or raises Exception: Model is Not Ready.
    *   If the method does not exist, the backend defaults to `True`.

*(Note: A third "Heartbeat" check using shared memory locks was considered for deadlock detection but is currently disabled to ensure stability in high-load CI environments.)*

## 3. Usage for Users

To utilize the custom readiness check, implement the `is_model_ready` method in your `model.py`:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.db_client = DatabaseClient("...")

    def execute(self, requests):
        # ... inference logic ...
        pass

    def is_model_ready(self):
        """
        Optional. Called when the server receives a readiness probe.
        Returns True if the model is ready to serve requests.
        """
        if not self.db_client.is_connected():
            return False
        return True
```

## 4. Testing & Verification

A dedicated test suite has been added to verify these behaviors.

### Location
`server/qa/L0_backend_python/model_ready_check/`

### Test Scenarios
The `test.sh` script verifies the following:
1.  **Initialization**: Starts Triton with a standard Python model.
2.  **Positive Case**: Verifies the model reports `READY` via both HTTP and gRPC.
3.  **Crash Detection (SIGSEGV)**:
    *   Sends `kill -11` to the stub process.
    *   Verifies the model immediately reports `NOT READY`.
    *   Verifies the server logs contain the specific error: `"Stub process ... is not healthy"`.
4.  **Force Kill Detection (SIGKILL)**:
    *   Sends `kill -9` to the stub process.
    *   Verifies the model immediately reports `NOT READY`.

### Running Tests
```bash
cd server/qa/L0_backend_python/model_ready_check
bash -ex test.sh
```

## 5. Development Details

### Key Files Modified
*   `python_backend/src/python_be.cc`: Main entry point for the readiness check logic.
*   `python_backend/src/stub_launcher.cc`: Implements `StubActive()` using `waitpid`.
*   `python_backend/src/pb_stub.cc`: Implements the IPC handler for calling Python code.
*   `python_backend/src/ipc_message.h`: Defines the new IPC protocol messages.

### Known Limitations
*   **Unresponsive/Deadlock**: Currently, if the process is alive but stuck in an infinite loop (holding the GIL), the check might still pass (unless the user's `is_model_ready` logic catches it). Future improvements could re-enable the shared memory heartbeat check.

