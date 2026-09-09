# Security and trust

Agent Injector is a stdio MCP server for a trusted local operator and trusted MCP
clients. It is not a network service, a multi-user authorization layer or an OS
sandbox. Never expose its tool interface to untrusted clients.

Tasks can read model-visible local files. Explicitly authorized Bash/Edit/Write
operations can change files and execute programs with the local account's rights.
`AGENT_WORKING_ROOT` verifies the starting directory, including symlink resolution;
it cannot prevent every later filesystem access or a path changing after admission.
Use a dedicated account or container for work requiring stronger isolation.

Remote-provider task text and tool outputs are transmitted to that provider.
Review provider terms before sharing sensitive data. HTTPS is required except for
explicit loopback HTTP endpoints. The server does not validate a local listener's
identity; configure its API key when appropriate.

Children receive only the selected provider key plus allowed platform/network
settings. An isolated Claude configuration, safe startup mode and explicit tools
reduce unintended inherited behavior; administrator-managed Claude policies remain
in effect. The tool interface does not accept arbitrary commands, endpoints or keys.

Prompts go through stdin. Logs contain lifecycle metadata, not prompt bodies or
raw provider exceptions. Results and bounded stderr are available to the trusted
MCP client; exact configured keys of at least four characters are redacted there.
Redaction is best effort and does not detect every transformed or unrelated secret.
Results remain in process memory until expiration/eviction/restart. No server-side
persistent transcript store is created. Tool-created files remain on disk.

Timeout, output and queue limits reduce runaway work. Process groups are terminated
on Linux/macOS, but deliberately detached descendants can escape that ownership.
Cancellation cannot undo file changes or charges already incurred.

For a vulnerability, use GitHub's private vulnerability reporting for this
repository when available. If it is unavailable, open an issue requesting a private
contact channel without including exploit details, credentials or personal data.
Only the current maintained version receives fixes; dependency scans are snapshots,
not proof that no vulnerabilities exist.
