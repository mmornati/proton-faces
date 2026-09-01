# Reference

Deep dives on the internals: how the three containers fit together, every environment variable, the full REST API surface, the security model, and the usual troubleshooting recipes.

<div class="pf-cards" markdown>

<div class="pf-card" markdown>
### 🏗️ [Architecture](architecture.md)
Three containers, two SQLite writers, one shared WAL DB. The bridge is strictly read-only. Every byte that crosses the network boundary is shown.
</div>

<div class="pf-card" markdown>
### ⚙️ [Configuration](configuration.md)
Every environment variable, what it does, what the default is, and when to tune it.
</div>

<div class="pf-card" markdown>
### 📡 [REST API](api.md)
Every endpoint, request shape, response shape, and required role.
</div>

<div class="pf-card" markdown>
### 🔒 [Security & privacy](security-privacy.md)
What's on disk, what's not, how bcrypt + opaque bearer tokens work, and the read-only bridge contract.
</div>

<div class="pf-card" markdown>
### ❓ [FAQ](faq.md)
Common questions and answers.
</div>

<div class="pf-card" markdown>
### 🛠️ [Troubleshooting](troubleshooting.md)
Common gotchas and how to fix them.
</div>

</div>

## A note on the docs vs the code

The docs are hand-curated from the source — they're not auto-generated. If you spot a drift between the docs and the code, please open an issue.
