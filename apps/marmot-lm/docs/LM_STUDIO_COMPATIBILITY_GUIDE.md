# LM Studio SDK Compatibility Guide (marmot-lm)

This guide describes the LM Studio WebSocket API compatibility layer implemented by `marmot-lm`.

**Source of truth:** `docs/lmstudio-sdk/` (vendored LM Studio SDK). If this guide and the SDK diverge, the SDK takes precedence.

## Quick Start

1. **Discovery** -- used by `LMStudioClient` to verify a server:

```http
GET /lmstudio-greeting
-> { "lmstudio": true }
```

2. **WebSocket connection:**

- Connect to `ws://localhost:<port>/<namespace>`
- Supported namespaces: `llm`, `embedding`, `system`, `diagnostics`, `files`, `repository`, `plugins`, `runtime`
- `/ws` is a legacy alias for `/llm`

3. **Authentication handshake** -- required as the first message on every WebSocket connection:

Client sends:
```json
{ "authVersion": 1, "clientIdentifier": "...", "clientPasskey": "..." }
```

Server responds:
```json
{ "success": true }
```

## Transport Protocol (SDK-Compatible)

The transport message shapes are defined in:
`docs/lmstudio-sdk/packages/lms-communication/src/Transport.ts`

### IDs

All IDs are integers, scoped to a single WebSocket connection:

| ID | Purpose |
|---|---|
| `callId` | RPC call/response correlation |
| `channelId` | Channel lifecycle tracking |
| `subscribeId` | Signal subscriptions |
| `ackId` | Optional per-message acknowledgement |

### Serialization (`SerializedOpaque`)

The SDK treats `parameter`, `creationParameter`, `message`, `patches[*]`, and `result` as `SerializedOpaque` (`any`).

- **`"raw"` serialization** (default): values are sent as normal JSON.
- **`"superjson"` serialization**: values are sent as SuperJSON payloads.
  - Example (top-level Map): `{ "json": [...], "meta": { "values": ["map"] } }`

### RPC

Request-response pattern for stateless operations.

Client sends:
```json
{ "type": "rpcCall", "endpoint": "version", "callId": 1, "parameter": null }
```

Server responds:
```json
{ "type": "rpcResult", "callId": 1, "result": { "version": "...", "build": 1 } }
```

For `void`/`undefined` endpoints, `result` may be omitted (the SDK accepts missing or `undefined` values).

### Channels

Bidirectional streaming pattern for long-running operations (model loading, inference, downloads).

Client opens a channel:
```json
{ "type": "channelCreate", "endpoint": "predict", "channelId": 2, "creationParameter": { ... } }
```

Server streams data to client:
```json
{ "type": "channelSend", "channelId": 2, "message": { "type": "fragment", "fragment": { ... } } }
```

Client sends control messages (e.g., cancellation):
```json
{ "type": "channelSend", "channelId": 2, "message": { "type": "cancel" } }
```

Server closes the channel:
```json
{ "type": "channelClose", "channelId": 2 }
```

### Errors

Transport-level errors (`rpcError`, `channelError`, `signalError`) use the SDK "extended error" shape defined in:
`docs/lmstudio-sdk/packages/lms-shared-types/src/Error.ts`

Fields:
- `title` (required)
- `cause`, `suggestion`, `errorData`, `displayData`, `stack`, `rootTitle` (all optional)

## Namespace Coverage

Backend interface definitions live in:
`docs/lmstudio-sdk/packages/lms-external-backend-interfaces/src/*BackendInterface.ts`

### `system`

- **RPC:** `listDownloadedModels`, `listDownloadedModelVariants`, `notify`, `version`, `setExperimentFlag`, `getExperimentFlags`, `startHttpServer`, `stopHttpServer`, `info`, `requestShutdown`
- **Channel:** `alive`

### `llm`

Base model endpoints:
- **Channel:** `loadModel`, `getOrLoad`
- **RPC:** `unloadModel`, `listLoaded`, `getModelInfo`, `getLoadConfig`, `getBasePredictionConfig`, `getInstanceProcessingState`, `estimateModelUsage`

LLM-specific endpoints:
- **Channel:** `predict`
- **RPC:** `applyPromptTemplate`, `tokenize`, `countTokens`, `preloadDraftModel`

### `embedding`

Base model endpoints:
- **Channel:** `loadModel`, `getOrLoad`
- **RPC:** `unloadModel`, `listLoaded`, `getModelInfo`, `getLoadConfig`, `getBasePredictionConfig`, `getInstanceProcessingState`, `estimateModelUsage`

Embedding-specific endpoints:
- **RPC:** `embedString`, `tokenize`, `countTokens`

### `diagnostics`

- **Channel:** `streamLogs` (client sends `{ "type": "stop" }` to end)

### `files`

- **RPC:** `getLocalFileAbsolutePath`, `uploadFileBase64`, `getDocumentParsingLibrary`
- **Channel:** `retrieve` (client sends `{ "type": "stop" }`), `parseDocument` (client sends `{ "type": "cancel" }`)

### `repository`

- **RPC:** `searchModels`, `getModelDownloadOptions`, `installPluginDependencies`, `getLocalArtifactFiles`, `loginWithPreAuthenticatedKeys`, `installLocalPlugin`, `getModelCatalog`
- **Channel:** `downloadModel`, `downloadArtifact`, `pushArtifact`, `ensureAuthenticated`, `createArtifactDownloadPlan`

### `plugins`

- **RPC:** `reindexPlugins`, `processingHandleUpdate`, `processingHandleRequest`, `processingPullHistory`, `processingGetOrLoadTokenSource`, `processingHasStatus`, `processingNeedsNaming`, `processingSuggestName`, `processingSetSenderName`, `setConfigSchematics`, `setGlobalConfigSchematics`, `pluginInitCompleted`
- **Channel:** `startToolUseSession`, `generateWithGenerator`, `registerDevelopmentPlugin`, `setPromptPreprocessor`, `setPredictionLoopHandler`, `setToolsProvider`, `setGenerator`

### `runtime`

- **RPC:** `listEngines`, `getEngineSelections` (superjson), `selectEngine`, `removeEngine`, `surveyHardware`, `searchRuntimeExtensions`
- **Channel:** `downloadRuntimeExtension`

## Implementation Notes (marmot-lm)

- **Connection-scoped IDs:** `marmot-lm` scopes channel and signal bookkeeping by `(connectionId, id)` to match `ClientPort` behavior, where IDs start at 1 per socket.
- **Channel cancellation:** A `channelSend` message with `message.type` set to `"cancel"`, `"stop"`, `"cancelDownload"`, `"cancelPlan"`, `"discardSession"`, or `"end"` triggers best-effort cancellation and channel closure.
- **Partial namespace coverage:** Some namespaces (`repository`, `plugins`, `runtime`, `files`, `diagnostics`) are intentionally minimal or no-op in areas not needed for local inference workflows. Responses still conform to SDK schemas.

## Code Map

| Area | Path |
|---|---|
| Protocol messages | `src/protocol/messages.rs` |
| WebSocket, auth, dispatch | `src/server/websocket.rs` |
| RPC handlers | `src/server/rpc/` |
| Channel handlers | `src/server/channels/` |
| Signals | `src/server/signals.rs` |
| SDK reference | `docs/lmstudio-sdk/` |
