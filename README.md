# Geo-Assist

Ask plain-English questions about your engineering documents and get answers with
citations — source file and page number — pulled straight out of the documents you
uploaded. It will not make things up: every answer is grounded in retrieved text.

**Nothing leaves the machine.** No cloud APIs, no telemetry, no external model
endpoints. The language models run locally on the server.

---

## Read this first

Geo-Assist is **already installed and running** on a Linux VM inside the network.
Nobody needs to install anything to use it.

This document is the operations manual for that VM. It is written to be followed
literally — every command can be copied and pasted as-is, with only the username
and IP address changed.

It has two halves:

- **[For users](#for-users)** — how to connect from your laptop and use the tool. Sections 1–6. No Linux knowledge needed.
- **[For administrators](#for-administrators)** — how to keep it running, change the model, update the code, and fix it when it breaks. Sections 7–20.

If you only ever do one thing with this document, do
[section 2](#2-connect-open-the-tunnel) — that's how you get in.

---

## Table of contents

**For users**

1. [What you need before you start](#1-what-you-need-before-you-start)
2. [Connect: open the tunnel](#2-connect-open-the-tunnel)
3. [Open Geo-Assist in your browser](#3-open-geo-assist-in-your-browser)
4. [Using Geo-Assist](#4-using-geo-assist)
5. [Disconnecting](#5-disconnecting)
6. [Connection troubleshooting](#6-connection-troubleshooting)

**For administrators**

7. [Logging in to the VM](#7-logging-in-to-the-vm)
8. [How the deployment is laid out](#8-how-the-deployment-is-laid-out)
9. [Command quick reference](#9-command-quick-reference)
10. [Everyday Docker commands](#10-everyday-docker-commands)
11. [Editing `docker-compose.yml`](#11-editing-docker-composeyml)
12. [Changing the chat model](#12-changing-the-chat-model)
13. [Changing the context length](#13-changing-the-context-length)
14. [Downloading a new model into Ollama](#14-downloading-a-new-model-into-ollama)
15. [Updating to a new version of the code](#15-updating-to-a-new-version-of-the-code)
16. [All configuration knobs](#16-all-configuration-knobs)
17. [Ingesting from a NAS share](#17-ingesting-from-a-nas-share)
18. [Backups and what you must not lose](#18-backups-and-what-you-must-not-lose)
19. [Server troubleshooting](#19-server-troubleshooting)
20. [Routine maintenance checklist](#20-routine-maintenance-checklist)

**Appendices**

- [A. Running Geo-Assist on a laptop instead](#appendix-a-running-geo-assist-on-a-laptop-instead)
- [B. Development, tests, and releases](#appendix-b-development-tests-and-releases)
- [C. How it works, briefly](#appendix-c-how-it-works-briefly)

---

# For users

## 1. What you need before you start

Three things — write them down somewhere you'll find them again:

| Thing | Looks like | Notes |
|---|---|---|
| **Username** | `organization_name` | Your login on the VM. Often not the same as your email or Windows username |
| **Server IP** | `VM-IP_address`| The VM's address on the network |
| **How you log in** | the VM's password |

You also need a terminal. You already have one:

- **Windows 10/11** — Start menu → type "PowerShell" → open it. SSH is built in.
- **macOS** — ⌘-Space → type "Terminal" → open it.
- **Linux** — your usual terminal.

You do **not** need Docker, Python, Ollama, or the Geo-Assist source code on your
laptop. All of that lives on the server.

---

## 2. Connect: open the tunnel

Geo-Assist listens on port **8743**, but only on the server's own loopback address
— it is deliberately not published to the network, so the document corpus can't be
reached by anyone who happens to be on the same LAN. An **SSH tunnel** forwards
port 8743 on *your* laptop through to port 8743 on *the server*, so your browser
can reach it as if it were running locally.

Paste this into your terminal, substituting your username and the server IP:

```bash
ssh -N -L 8743:127.0.0.1:8743 organization_name@ip_address
```

Enter your password when prompted. **Then nothing will happen** — no output, no new
prompt, just a cursor sitting there. That is correct. That is what success looks
like.

### What each part means

| Part | Meaning |
|---|---|
| `ssh` | Start a secure connection |
| `-N` | Don't open a shell — this connection is only for forwarding |
| `-L 8743:127.0.0.1:8743` | Forward **my** port 8743 → the server's `127.0.0.1:8743` |
| `organization_name@ip_address` | Who I am and which machine I'm connecting to |

### Leave that window open

The tunnel exists only while that terminal window is running. Close it, press
`Ctrl+C`, or let the laptop sleep, and Geo-Assist stops loading in your browser.
Minimise the window instead of closing it.

## 3. Open Geo-Assist in your browser

With the tunnel running, go to:

**<http://localhost:8743>**

Bookmark it. If the page doesn't load, see
[Connection troubleshooting](#6-connection-troubleshooting).

---

## 4. Using Geo-Assist

### Ask a question

Type it in the box at the bottom and press Enter. Ask the way you'd ask a
colleague — full sentences work far better than keywords.

Answers stream in as they're generated. Below each answer are the source files and
page numbers it used; click one to open the original document.

The first question after an idle period is slower — the model has to load into
memory first. Subsequent questions are quicker.

You can ask in **English, Armenian, or Russian**; it replies in the language you
asked in.

### Upload documents

Click **Upload** in the sidebar and pick files, or drag a whole folder onto the
window. Dropping a folder collects every file inside it recursively and pre-fills
the folder name.

Supported: `.pdf`, `.docx`, `.pptx`, `.txt`, `.csv`

A progress bar tracks ingestion. Large batches take a while — text extraction and
embedding both run on the server's CPU. You can keep asking questions while it
runs.

Re-uploading a file that's already in the system is harmless: documents are
identified by their content, so nothing gets duplicated.

### Folders

Documents can be filed into folders, moved between them, and questions can be
restricted to a single folder. Useful when one project's specs shouldn't bleed into
another project's answers.

## Scan the QNAP

If the company QNAP is connected, the sidebar has a Scan NAS folder button. Use
it instead of uploading when the documents already live on the share — nothing gets
copied or moved, and you don't have to download anything first.

Pick a subfolder, click Preview to see how many files would be ingested, then
Start. Progress shows in the same bar as a normal upload.

Preview first, every time. It's the difference between ingesting one project folder
and ingesting the entire share.

Geo-Assist can never write to the QNAP. The share is mounted read-only — it can
read your documents and nothing else. Scanning cannot move, rename, or delete
anything.

If the button isn't there, the share isn't connected on the server. That's an
administrator job, not something you can fix from the browser.

### Procedure mode

Hover over a document in the sidebar and click the **▶** button. Geo-Assist reads
the document and walks you through it one step at a time. If a step conflicts with
something in another ingested document, it flags it with **⚠️ CONFLICT**.

### Deleting

Remove a single document from its entry in the sidebar, or use **Clear all** to
wipe the index entirely.

**Clear all is not undoable.** Everything has to be re-uploaded and re-embedded,
which on a large corpus is hours.

---

## 5. Disconnecting

Go to the terminal window running the tunnel and press **`Ctrl+C`**. That's it.
Nothing on the server is affected — Geo-Assist keeps running for everyone else, and
the documents stay indexed.

---

## 6. Connection troubleshooting

### `Connection refused` or `Operation timed out`

You can't reach the server at all. Check you're on the right network (VPN
connected, if one is required), that the IP is right, and that the VM is powered on.

### `Permission denied (publickey,password)`

Wrong username or wrong password. The username is the VM's login,
which may not match your Windows or email username.

### `bind [127.0.0.1]:8743: Address already in use`

Something on your laptop is already using port 8743 — almost always a tunnel you
opened earlier and forgot. Either find and close that window, or forward to a
different local port:

```bash
ssh -N -L 8800:127.0.0.1:8743 organization_name@ip_address
```

...then browse to <http://localhost:8800> instead.

### The terminal shows nothing after I enter my password

That's success. `-N` means "no shell". Leave it running and open your browser.

### Browser says "This site can't be reached"

In order:

1. Is the tunnel window still open and running? (Not closed, not `Ctrl+C`'d, laptop not asleep.)
2. Are you using `http://`, not `https://`? There's no TLS on the tunnel — SSH already encrypts it.
3. Is it `localhost:8743`, not the server's IP? You browse to **your own** machine; SSH does the rest.
4. If all of that is right, the app on the server is probably down. See [section 19](#19-server-troubleshooting), or tell whoever administers it.

### It worked, then stopped after I went to lunch

The tunnel timed out. Press `Ctrl+C` and run the command again. Add
`ServerAliveInterval 30` to `~/.ssh/config` (see
[section 2](#make-it-a-one-word-command-recommended)) to prevent it.

### The page loads but every answer says it can't find anything

Either the index was cleared, or your question is about a document nobody uploaded.
Check the sidebar — if the document list is empty, that's your answer.

---

# For administrators

Everything below assumes you're logged in to the VM over SSH.

> **The VM user is not in the `docker` group.** Every `docker` command below needs
> `sudo`. Forget it and you get
> `permission denied while trying to connect to the Docker daemon socket`.

## 7. Logging in to the VM

Same command as the tunnel, without `-N -L` — this time you want an actual shell:

```bash
ssh organization_name@ip_address
```

You can have both at once: one window holding the tunnel, another with a shell. Or
do both in one connection by dropping `-N`:

```bash
ssh -L 8743:127.0.0.1:8743 organization_name@ip_address
```

Almost every command in this half starts from the deployment directory, so get in
the habit:

```bash
cd /opt/geo-assist
```

---

## 8. How the deployment is laid out

| | |
|---|---|
| Compose file | `/opt/geo-assist/docker-compose.yml` |
| Application source | `/opt/geo-assist/app` (the Git checkout) |
| Compose project name | `geo-assist` |
| Containers | `geo-assist-app`, `geo-assist-ollama`, `geo-assist-qdrant` |
| App port | `8743`, bound to loopback only |
| Persistent data | `/opt/geo-assist/data` on the host → `/app/data` in the container |

Three containers:

- **`geo-assist-app`** — the application and the web UI. This is what port 8743 reaches.
- **`geo-assist-ollama`** — runs the chat and embedding models. Everything about models happens in here.
- **`geo-assist-qdrant`** — the vector database holding the embedded document chunks.

Four things about this layout that will bite you if you don't know them:

**The compose file and the app `Dockerfile` are not in the Git repository.** The VM
holds the only copy. Read the real file on the box rather than trusting this
README's examples, and keep a backup off the box — see
[section 18](#18-backups-and-what-you-must-not-lose).

**The app image bakes in the source code.** Copying a `.py` file onto the VM does
nothing on its own. Shipping a code change means rebuilding the image — see
[section 15](#15-updating-to-a-new-version-of-the-code).

**The `./data:/app/data` bind mount is load-bearing.** Remove it and every rebuild
destroys the original uploaded files, permanently.

**Port 8743 is bound to `127.0.0.1` on purpose.** The SSH tunnel is the access
method. Publishing it to `0.0.0.0` would expose the entire document corpus to
anyone on the network.

---

## 9. Command quick reference

The ten commands that cover ninety percent of what you'll ever need. All are run
from `/opt/geo-assist`.

```bash
cd /opt/geo-assist

sudo docker compose ps                    # is everything running?
sudo docker compose logs -f app           # watch the app's logs (Ctrl+C to stop)
sudo docker compose restart app           # restart the app
sudo docker compose up -d                 # apply compose-file changes / start everything
sudo docker compose down                  # stop everything (never add -v)

sudo nano /opt/geo-assist/docker-compose.yml     # edit config  (Ctrl+O, Enter, Ctrl+X)
sudo docker compose config                       # validate that edit before applying

sudo docker exec -it geo-assist-ollama ollama list                # what models exist
sudo docker exec -it geo-assist-ollama ollama pull (model name)   # download a model

cd /opt/geo-assist/app && sudo git pull && cd .. \
  && sudo docker compose build app && sudo docker compose up -d app   # update the code
```

---

## 10. Everyday Docker commands

| Task | Command |
|---|---|
| What's running? | `sudo docker compose ps` |
| Follow the app's logs | `sudo docker compose logs -f app` |
| Last 200 lines of logs | `sudo docker compose logs --tail 200 app` |
| Logs for Ollama | `sudo docker compose logs -f ollama` |
| Restart just the app | `sudo docker compose restart app` |
| Apply a compose-file change | `sudo docker compose up -d` |
| Stop everything | `sudo docker compose down` |
| Start everything | `sudo docker compose up -d` |
| Disk used by Docker | `sudo docker system df` |
| Shell inside the app container | `sudo docker exec -it geo-assist-app bash` |

`sudo docker compose up -d` is the workhorse: it compares the running containers
against the compose file and recreates only what changed. **`restart` does not pick
up edits to `docker-compose.yml`** — use `up -d` for those.

### Reading the logs

The app's logs are where slow answers get diagnosed. Every Ollama call logs a line
like:

```
ollama qwen3.5:4b: load=0.0s prompt=1840 tok/12.3s eval=310 tok/18.9s thinking=0 chars
```

| Field | If it's large |
|---|---|
| `load` | The model was cold and had to be loaded from disk |
| `prompt` | The assembled context is oversized — see [section 13](#13-changing-the-context-length) |
| `eval` | Normal — this is the answer being generated |
| `thinking` | The model is burning time on hidden reasoning that never gets shown. `"think": false` isn't taking effect for that model |

Without these numbers, a six-minute answer is indistinguishable from a hung
process. They are the first thing to look at when someone reports slowness.

---

## 11. Editing `docker-compose.yml`

```bash
sudo nano /opt/geo-assist/docker-compose.yml
```

### nano in thirty seconds

`nano` is a plain text editor that runs in the terminal. There's no mouse — arrow
keys move the cursor.

| Key | Does |
|---|---|
| `Ctrl+O` then `Enter` | **Save** (it calls this "Write Out") |
| `Ctrl+X` | Exit (prompts to save if you haven't) |
| `Ctrl+W` | Search for text |
| `Ctrl+K` | Cut the current line |
| `Ctrl+U` | Paste the cut line |
| `Ctrl+_` | Jump to a line number |
| `Ctrl+G` | Help |

The `^` in nano's bottom bar means Ctrl — so `^O` is `Ctrl+O`.

### Back it up before you touch it

This file exists nowhere else:

```bash
sudo cp /opt/geo-assist/docker-compose.yml /opt/geo-assist/docker-compose.yml.bak
```

### YAML will punish a stray space

Indentation is meaningful and **tabs are illegal** — spaces only. Keep new lines
aligned with their siblings. Always check before applying:

```bash
sudo docker compose config
```

That prints the fully-resolved file, or an error with a line number if the YAML is
malformed. Run it every single time, before `up -d`.

### What the file looks like

Roughly this shape — **read the real file on the VM; don't assume it matches**:

```yaml
services:
  app:
    build: ./app
    container_name: geo-assist-app
    ports:
      - "127.0.0.1:8743:8743"        # loopback only — the SSH tunnel is the way in
    environment:
      - GEO_OLLAMA_BASE=http://ollama:11434
      - GEO_QDRANT_HOST=qdrant
      - GEO_CHAT_MODEL=qwen3.5:9b     # ← the knob you'll change most often
      - GEO_KEEP_ALIVE=5m
    volumes:
      - ./data:/app/data              # ← DO NOT REMOVE. See section 18.
    depends_on:
      - ollama
      - qdrant
    restart: unless-stopped

  ollama:
    image: ollama/ollama
    container_name: geo-assist-ollama
    environment:
      - OLLAMA_MAX_LOADED_MODELS=2    # keep chat + embed models both resident
      - OLLAMA_KEEP_ALIVE=5m
    volumes:
      - ollama-models:/root/.ollama   # downloaded models live here
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant
    container_name: geo-assist-qdrant
    environment:
      - QDRANT__TELEMETRY_DISABLED=true
    volumes:
      - ./data/qdrant:/qdrant/storage
    restart: unless-stopped

volumes:
  ollama-models:
```

### Applying an edit

```bash
cd /opt/geo-assist
sudo docker compose config          # validate
sudo docker compose up -d           # apply
sudo docker compose ps              # confirm they came back up
```

### Undoing an edit that broke something

```bash
sudo cp /opt/geo-assist/docker-compose.yml.bak /opt/geo-assist/docker-compose.yml
sudo docker compose up -d
```

---

## 12. Changing the chat model

The chat model is set by `GEO_CHAT_MODEL` on the **app** service. It currently
defaults to `qwen3.5:9b`.

**The model must already be downloaded into Ollama** — see
[section 14](#14-downloading-a-new-model-into-ollama). Pointing the app at a model
Ollama doesn't have produces an error on every question.

Full sequence:

```bash
# 1. Pull the model first
sudo docker exec -it geo-assist-ollama ollama pull (model name)

# 2. Back up, then edit
sudo cp /opt/geo-assist/docker-compose.yml /opt/geo-assist/docker-compose.yml.bak
sudo nano /opt/geo-assist/docker-compose.yml
```

Find the app service's `environment:` block and change the line:

```yaml
      - GEO_CHAT_MODEL=(model name)
```

Save (`Ctrl+O`, `Enter`), exit (`Ctrl+X`), then:

```bash
cd /opt/geo-assist
sudo docker compose config
sudo docker compose up -d app
sudo docker compose logs -f app
```

Ask a question in the browser to confirm — the logs name the model on every call.

### Which model to pick

Before swapping the default permanently, run the benchmark. It replays identical
retrieved context to every model, so the numbers reflect the model rather than
retrieval jitter:

```bash
sudo docker exec -it geo-assist-app python3 benchmark_models.py \
    --models qwen3.5:4b llama3.1:8b --runs 3
```

The column that matters is **numeric drift** — whether the same question yields
"28 V" once and "24 V" the next time. A model with non-zero drift is unusable for
spec lookups no matter how fast it is. Treat near-zero drift as the entry
requirement and speed as the tiebreak, not the reverse.

### If you switch to a reasoning model

Qwen-style reasoning models must be called with `"think": false` or they silently
spend 20–30 s per answer generating hidden reasoning before any visible output
appears. This is already applied at every Ollama call site in `llm.py`. If the logs
show `thinking=NNNN chars`, it isn't taking effect for that model and the latency is
real — pick a different model or accept the delay.

### Changing the *embedding* model is a much bigger job

`GEO_EMBED_MODEL` (default `nomic-embed-text`) determines the vector space every
stored chunk lives in. Changing it invalidates the entire index: you must also set
`GEO_EMBED_DIM` to the new model's dimension and **re-ingest every document**. Don't
change it casually, and don't change it at all without a maintenance window.

---

## 13. Changing the context length

The context length is how much text the model can consider at once — retrieved
document chunks, the system prompt, the conversation history, and the answer, all
together. Ollama's default is small (4096 tokens), and anything past it is silently
truncated from the front. That shows up as the model appearing to ignore documents
you can see listed in its own citations.

It's an **Ollama-side** setting, not an app one. Set it on the **ollama** service:

```bash
sudo nano /opt/geo-assist/docker-compose.yml
```

```yaml
  ollama:
    environment:
      - OLLAMA_CONTEXT_LENGTH=8192
```

```bash
cd /opt/geo-assist
sudo docker compose config
sudo docker compose up -d ollama
```

### Picking a number

| Value | When |
|---|---|
| `4096` | Ollama's default. Fine for short questions over small documents |
| `8192` | Sensible general setting for this corpus |
| `16384` | Long tables and multi-document comparisons. Noticeably slower on CPU |
| `32768`+ | Only with a GPU or a lot of spare RAM |

**Longer context is not free.** Memory use scales with it, and on CPU-only hardware
prompt evaluation scales roughly linearly — doubling the context can double the
wait before the first token appears. On a 16 GB box, `8192` is a sensible ceiling
for routine use.

Check what you're actually spending:

```bash
sudo docker compose logs --tail 200 app | grep prompt=
```

A `prompt=13480 tok/573.0s` line is the failure mode — nine and a half minutes of
prompt evaluation before the model said a word. If you see numbers in that range,
the fix is almost never "raise the context": it's oversized *retrieved* context,
which is bounded deliberately in `retriever.py`. Don't remove those bounds without
re-measuring.

### The app-side knobs that decide how much context gets used

These live on the **app** service and control how much text gets assembled in the
first place:

| Variable | Default | Effect |
|---|---|---|
| `GEO_RETRIEVAL_K` | `15` | Candidate chunks fetched per query before fusion. Don't go above ~20 |
| `GEO_QUERY_EXPANSION` | `false` | Generates up to 3 query variants. Better cross-document recall, costs a full extra LLM round-trip (~20–25 s on CPU) before retrieval even begins |
| `GEO_RERANK` | `true` | Cross-encoder rescoring of the top 20 candidates. Big accuracy win; falls back silently if the model isn't installed |

If answers are missing information that's clearly in the documents, try
`GEO_QUERY_EXPANSION=true` before reaching for a bigger context window.

---

## 14. Downloading a new model into Ollama

Ollama runs **inside** the `geo-assist-ollama` container, so you run `ollama`
commands *through* that container with `docker exec`. There is no `ollama` binary on
the VM itself — typing `ollama pull` at the VM's shell will just say
`command not found`.

The pattern is always:

```
sudo docker exec -it geo-assist-ollama ollama <whatever>
```

### Pull a model

```bash
sudo docker exec -it geo-assist-ollama ollama pull (model name)
```

You'll see a progress bar per layer. A 4B model is a few GB; an 8B model is larger.
The VM needs internet access for this.

Browse what's available at <https://ollama.com/library>. The tag after the colon is
the size or variant — `qwen3.5:4b`, `llama3.1:8b`. No tag means `:latest`.

### List what's installed

```bash
sudo docker exec -it geo-assist-ollama ollama list
```

```
NAME                    ID              SIZE      MODIFIED
qwen3.5:4b              a1b2c3d4e5f6    2.6 GB    3 weeks ago
nomic-embed-text:latest f0e1d2c3b4a5    274 MB    3 weeks ago
```

### Delete a model you're not using

Models are large and the VM's disk is not. Remove the ones you've finished
benchmarking:

```bash
sudo docker exec -it geo-assist-ollama ollama rm llama3.1:8b
```

**Never remove `nomic-embed-text`** — every ingest and every query needs it, and
removing it breaks the system until it's pulled again.

### See what's loaded in memory right now

```bash
sudo docker exec -it geo-assist-ollama ollama ps
```

### Test a model directly, outside the app

```bash
sudo docker exec -it geo-assist-ollama ollama run qwen3.5:9b "Say hello in one word."
```

Useful for isolating whether a problem is the model or the retrieval pipeline. Type
`/bye` to leave an interactive session.

### Where the downloads go

Into a named Docker volume (`ollama-models` in the example above), mounted at
`/root/.ollama` inside the container. They survive `docker compose down`, container
recreation, and image rebuilds.

They do **not** survive `docker compose down -v`. That flag deletes volumes. Don't
use it.

### Air-gapped machines

`ollama pull` needs internet. On a genuinely isolated box, pull the model on a
connected machine and copy the `~/.ollama/models` directory across. Plan this
*before* disconnecting the VM, not after.

### After pulling

Downloading a model doesn't switch to it. Set `GEO_CHAT_MODEL` and restart the app
— [section 12](#12-changing-the-chat-model).

---

## 15. Updating to a new version of the code

The app image **bakes in the source code** at build time. That means pulling new
code is a two-step job: get the new source onto the VM, then rebuild the image from
it. Doing only one of the two changes nothing.

The Git checkout lives at `/opt/geo-assist/app` (the `build: ./app` path in the
compose file). Confirm before you start:

```bash
cd /opt/geo-assist/app
sudo git status
sudo git remote -v
```

### The standard update

```bash
# 1. Back up the compose file — it is not in Git
sudo cp /opt/geo-assist/docker-compose.yml /opt/geo-assist/docker-compose.yml.bak

# 2. Note the version you're on now, so you can go back
cd /opt/geo-assist/app
sudo git rev-parse --short HEAD          # write this down

# 3. Pull the new code
sudo git pull

# 4. Rebuild the image and restart the app
cd /opt/geo-assist
sudo docker compose build app
sudo docker compose up -d app

# 5. Watch it come up
sudo docker compose logs -f app
```

Then open <http://localhost:8743> through the tunnel and ask a question you know
the answer to.

`build` alone doesn't swap the running container. `up -d` alone doesn't pick up new
source. You need both, in that order.

### Updating to a specific released version

```bash
cd /opt/geo-assist/app
sudo git fetch --all --tags
sudo git checkout v1.2.0
cd /opt/geo-assist
sudo docker compose build app
sudo docker compose up -d app
```

### Rolling back

This is why you wrote down the commit hash. If the new version misbehaves:

```bash
cd /opt/geo-assist/app
sudo git checkout <the-hash-you-wrote-down>
cd /opt/geo-assist
sudo docker compose build app
sudo docker compose up -d app
```

Rolling back the *code* does not roll back the *data*. If an update changed how
documents are stored, the index may need rebuilding — which is why you take a data
backup first ([section 18](#18-backups-and-what-you-must-not-lose)) for anything
beyond a routine patch.

### If `git pull` refuses because of local changes

Someone edited a file on the VM directly. See what changed before destroying it:

```bash
cd /opt/geo-assist/app
sudo git diff
```

If those changes matter, save them:

```bash
sudo git stash
sudo git pull
sudo git stash pop        # may need conflicts resolved by hand
```

If they don't matter, discard them:

```bash
sudo git checkout -- .
sudo git pull
```

### If the VM has no internet access

`git pull` needs to reach the remote. On an isolated VM, produce an archive on a
connected machine and copy it across:

```bash
# on a connected machine, inside the repo
git archive --format=tar.gz -o geo-assist-v1.2.0.tar.gz v1.2.0

# copy it over, then on the VM
sudo tar xzf geo-assist-v1.2.0.tar.gz -C /opt/geo-assist/app
cd /opt/geo-assist
sudo docker compose build app
sudo docker compose up -d app
```

Note that `docker compose build` itself may need to fetch Python packages. If the
VM is fully offline, the image has to be built elsewhere and loaded with
`docker save` / `docker load`.

### A checklist for anyone doing this the first time

1. [ ] Back up `docker-compose.yml`
2. [ ] Record the current commit hash
3. [ ] `git pull` in `/opt/geo-assist/app`
4. [ ] `sudo docker compose build app`
5. [ ] `sudo docker compose up -d app`
6. [ ] `sudo docker compose ps` — all three containers `Up`
7. [ ] `sudo docker compose logs --tail 50 app` — no tracebacks
8. [ ] Open the browser through the tunnel, ask a known question, check the citations still resolve

---

## 16. All configuration knobs

Set as environment variables on the **app** service in `docker-compose.yml` (except
`OLLAMA_*`, which go on the ollama service). Defaults live in `config.py`.

### Models and memory

| Variable | Default | Purpose |
|---|---|---|
| `GEO_CHAT_MODEL` | `qwen3.5:9b` | Ollama chat model |
| `GEO_EMBED_MODEL` | `nomic-embed-text` | Embedding model. Changing it requires a full re-index |
| `GEO_EMBED_DIM` | `768` | Must match the embedding model's dimension |
| `GEO_EXPAND_MODEL` | same as chat model | Model used for query expansion |
| `GEO_KEEP_ALIVE` | `5m` | How long Ollama holds a model in RAM after last use. `-1` = never unload (fast first query, memory-hungry), `0` = unload immediately |
| `GEO_OLLAMA_BASE` | `http://127.0.0.1:11434` | Ollama endpoint — in compose this points at the `ollama` service |

### Speed and throughput

| Variable | Default | Purpose |
|---|---|---|
| `GEO_EMBED_CONCURRENCY` | `2` | Parallel embedding requests. `4` with a GPU |
| `GEO_CHAT_CONCURRENCY` | `1` | Concurrent background generations. Ollama serialises on CPU, so >1 buys nothing and costs RAM |
| `GEO_PREPARE_CONCURRENCY` | `min(8, cores)` | Files parsed in parallel during ingest. **Lower this to `2` first if ingest is straining memory** |

### Retrieval quality

| Variable | Default | Purpose |
|---|---|---|
| `GEO_QUERY_EXPANSION` | `false` | Multi-query expansion. Better cross-document recall, +20–25 s per query |
| `GEO_RERANK` | `true` | Cross-encoder reranking. Needs the reranker model pre-downloaded; falls back silently if absent |
| `GEO_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |

### Images and OCR

| Variable | Default | Purpose |
|---|---|---|
| `GEO_OCR` | `false` | Extract text from images embedded in documents. Needs `requirements-ocr.txt` (~200 MB) in the image |
| `GEO_OCR_MIN_WORDS` | `10` | Minimum words for OCR output to count as a caption |
| `GEO_KEEP_IMAGES` | `true` | Keep images whose OCR was too sparse (diagrams, schematics), flagged for a future vision pass |

### Storage

| Variable | Default | Purpose |
|---|---|---|
| `GEO_QDRANT_HOST` | `127.0.0.1` | In compose, the `qdrant` service name |
| `GEO_QDRANT_PORT` | `6333` | Qdrant port |
| `GEO_QDRANT_INDEX` | `geo_docs` | Collection name |

### NAS

| Variable | Default | Purpose |
|---|---|---|
| `GEO_NAS_ROOT` | `/app/documents` | Share root as seen inside the container |
| `GEO_NAS_IO_ERROR_LIMIT` | `10` | Read failures before a scan aborts with "NAS unreachable" |

### Ollama service

| Variable | Suggested | Purpose |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | `8192` | Context window — see [section 13](#13-changing-the-context-length) |
| `OLLAMA_MAX_LOADED_MODELS` | `2` | Keeps chat and embedding models both resident, avoiding a ~30 s cold load on the first query |
| `OLLAMA_KEEP_ALIVE` | `5m` | Ollama's own residency timer |

---

## 17. Ingesting from a NAS share

Mount the share **read-only** on the host, bind-mount it into the app container at
`/app/documents`, and the sidebar grows a **Scan NAS folder** button. Users pick a
subfolder, hit Preview for a count of what would be ingested, then Start. Progress
reuses the normal ingest bar.

In `docker-compose.yml`, on the app service:

```yaml
    volumes:
      - ./data:/app/data
      - /mnt/nas-share:/app/documents:ro    # the :ro is not optional
```

The `:ro` means the app has no write path to the NAS regardless of what the code
does. Keep it.

Re-scans are cheap. `data/nas_manifest.db` records each file's path, size and
mtime, so unchanged files are skipped without being read — the difference between a
re-scan in seconds and one in hours over CIFS. Deleting that file forces a full
re-scan: safe, but slow.

NAS subfolders become document folders. A file already in the index is **skipped,
not rewritten**, so a document someone filed by hand into one folder keeps that
folder even if identical bytes also live on the share.

If the sidebar button never appears, check the health endpoint from inside the
container:

```bash
sudo docker exec -it geo-assist-app curl -s localhost:8743/ingest/nas/health
```

- `missing` — no bind mount
- `unreadable` — usually the container's UID not matching the CIFS mount's `uid=` option
- `ok` — working

The first two are otherwise indistinguishable from an empty share.

---

## 18. Backups and what you must not lose

### The compose file and the Dockerfile

Neither is in Git. **The VM holds the only copy.** Losing them loses the record of
how this system is actually deployed. Copy them off the box now, and again after
every edit:

```bash
# run this from your laptop
scp organization_name@ip_address:/opt/geo-assist/docker-compose.yml ~/backups/
```

### The `./data:/app/data` bind mount

Do not remove this line from `docker-compose.yml`. Without it, `data/originals/` —
the original uploaded files — is destroyed on every rebuild. That is
**unrecoverable**, and it permanently breaks the source link on every citation in
every existing answer.

### What's in `data/`

| Path | Contents | If lost |
|---|---|---|
| `data/originals/` | Original uploaded files | **Unrecoverable.** Citation links break permanently |
| `data/qdrant/` | Vector index | Rebuildable by re-ingesting everything (slow) |
| `data/bm25_index.pkl` | Keyword index | Rebuilt automatically on next start |
| `data/images/` | Extracted figures | Rebuildable by re-ingesting |
| `data/nas_manifest.db` | NAS scan state | Rebuildable — the next scan is just slow |

### Taking a full backup

With the containers stopped, a tarball of `data/` plus the compose file is
complete:

```bash
cd /opt/geo-assist
sudo docker compose stop
sudo tar czf ~/geo-assist-backup-$(date +%F).tar.gz data/ docker-compose.yml
sudo docker compose start
```

Then copy that tarball off the VM. A backup that only exists on the machine it's
backing up is not a backup.

### Never run `docker compose down -v`

`-v` deletes named volumes, which is where the downloaded Ollama models live.

---

## 19. Server troubleshooting

### Users can't reach it, but their tunnel command is right

```bash
cd /opt/geo-assist
sudo docker compose ps
```

Any container not showing `Up` is the problem. Check its logs, then:

```bash
sudo docker compose up -d
```

### Every question errors out

Usually Ollama. Check it's up and that the configured model actually exists:

```bash
sudo docker compose logs --tail 100 ollama
sudo docker exec -it geo-assist-ollama ollama list
```

If `GEO_CHAT_MODEL` names a model that isn't in that list, pull it
([section 14](#14-downloading-a-new-model-into-ollama)).

### Answers are extremely slow

```bash
sudo docker compose logs --tail 200 app | grep ollama
```

| Symptom | Cause | Fix |
|---|---|---|
| `load=30.0s` on every call | The model is being evicted between queries | Raise `GEO_KEEP_ALIVE`; set `OLLAMA_MAX_LOADED_MODELS=2` |
| `prompt=` in the thousands | Oversized context | [Section 13](#13-changing-the-context-length) |
| `thinking=` non-zero | Hidden reasoning tokens | `"think": false` isn't taking effect for that model |

### Ingest is running the machine out of memory

Lower `GEO_PREPARE_CONCURRENCY` to `2` on the app service. That's the first knob to
reach for on a 16 GB box.

### The VM is out of disk

```bash
df -h
sudo docker system df
sudo docker image prune -a       # removes unused images — safe
sudo docker exec -it geo-assist-ollama ollama list    # then rm what you don't need
```

Old model downloads and superseded app images are usually the culprits. Never prune
volumes.

### Search returns nothing for terms visible in a document

Qdrant may not have come up cleanly, or the index was cleared:

```bash
sudo docker compose logs --tail 100 qdrant
```

### After a reboot, nothing is running

The compose services use `restart: unless-stopped`, so they should come back on
their own. If they don't:

```bash
cd /opt/geo-assist
sudo docker compose up -d
```

---

# Appendices

## Appendix A. Running Geo-Assist on a laptop instead

The VM is the primary deployment and what everything above describes. Standalone
builds still exist for Windows, macOS and Linux and are what the release zips ship
— useful for a demo laptop or a genuinely disconnected machine. They are not what
anyone in the office should be using day to day.

Download the latest release for your OS from the [Releases page](../../releases):

| OS | What to do |
|---|---|
| Windows | Download `geo-assist-windows-*.zip`, unzip, double-click `start.bat` |
| macOS | Download `geo-assist-macos-*.zip`, unzip, run `./start_mac.sh` |
| Linux | Download `geo-assist-linux-*.zip`, unzip, run `./start_linux.sh` |

The start script detects your hardware, checks for Ollama and a C compiler
(offering to install anything missing via winget / brew / apt), pulls the required
models, starts a local Qdrant server bound to `127.0.0.1`, installs Python
dependencies, and opens <http://localhost:8743> when ready. On exit it stops what it
started — a pre-existing Qdrant instance is left alone.

**The Qdrant binary is not bundled** (~30 MB). Download it once from the
[Qdrant releases page](https://github.com/qdrant/qdrant/releases) and unpack it into
a `qdrant/` folder next to the start script:

| OS | Asset | Resulting path |
|---|---|---|
| Windows | `qdrant-x86_64-pc-windows-msvc.zip` | `qdrant\qdrant.exe` |
| macOS (Apple Silicon) | `qdrant-aarch64-apple-darwin.tar.gz` | `qdrant/qdrant` |
| macOS (Intel) | `qdrant-x86_64-apple-darwin.tar.gz` | `qdrant/qdrant` |
| Linux | `qdrant-x86_64-unknown-linux-gnu.tar.gz` | `qdrant/qdrant` |

On macOS and Linux make it executable — and on macOS clear the quarantine flag, or
Gatekeeper kills it silently and the start script just reports a timeout:

```bash
chmod +x qdrant/qdrant
xattr -dr com.apple.quarantine qdrant/qdrant   # macOS only
```

Override a setting before launching:

```powershell
$env:GEO_CHAT_MODEL = "qwen3.5:4b"
.\start.ps1
```

Manual run, any OS:

```bash
ollama pull qwen3.5:4b && ollama pull nomic-embed-text
pip3 install -r requirements.txt
python3 -m uvicorn main:app --host 127.0.0.1 --port 8743
```

Reference hardware for this path is a Dell Precision 3260 — 12th-gen Core i5/i7/i9,
no GPU, 16 GB RAM. `config.py`'s defaults are tuned for exactly that profile.

### Bulk re-indexing a local directory

```bash
python3 reindex.py --dir ~/Desktop/my-docs --folder "Project Alpha"
python3 reindex.py --dir ~/Desktop/my-docs --limit 50
```

**Only run this with the server stopped.** The BM25 keyword index is in-memory state
persisted to a pickle file; a second process that loads it, adds to it and writes it
back while the server holds a stale copy loses whichever write lands first. (This is
also why NAS ingestion is a route inside the server rather than a standalone
script.)

---

## Appendix B. Development, tests, and releases

```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/test_retriever.py -v
```

Tests run fully offline — no Ollama, no GPU, no running Qdrant server. Embeddings
and chat are mocked with deterministic fakes and the store runs in Qdrant's embedded
mode. `tests/test_airgap.py` pins the air-gap guarantees in code rather than leaving
them to deployment discipline.

They cover pipeline logic, not real-world speed or answer quality. For those, use
`benchmark_models.py` against a running instance with documents ingested.

Linting:

```bash
pip install ruff
ruff check .
ruff check --fix .
```

**Every new feature or bug fix must include tests.**

### Cutting a release

Push a tag matching `v*`:

```bash
git tag v1.1.0 && git push origin v1.1.0
```

`.github/workflows/release.yml` builds three trimmed zips (one per OS) via
`git archive` and publishes them to a GitHub Release. Dev-only paths — `tests/`,
`docs/`, `.github/`, `CLAUDE.md`, `handoff.md` — are excluded. Anyone needing the
full source should use GitHub's automatic per-tag "Source code" links.

Note that releases only affect the laptop install path. The VM is updated by
pulling code and rebuilding — [section 15](#15-updating-to-a-new-version-of-the-code).

---

## Appendix C. How it works, briefly

```
geo-assist/
├── main.py          FastAPI app + routes + static serving
├── config.py        All tunable constants + air-gap env kill-switches
├── llm.py           Async Ollama wrappers (embed + chat), trust_env=False
├── store.py         Qdrant/Haystack document store — the only DB seam
├── ingest.py        File parsing, chunking, embedding, storage
├── retriever.py     Hybrid RAG pipeline (semantic + BM25 + RRF + rerank)
├── bm25_index.py    Incremental BM25 index, persisted between restarts
├── nas.py           NAS share walker — junk exclusion, path containment
├── nas_manifest.py  SQLite record of which NAS files have been ingested
├── ingest_nas.py    NAS scan driver (walk → manifest diff → batch ingest)
├── static/
│   └── index.html   Single-file frontend, no build step (air-gap safe)
└── data/            Qdrant storage, originals, images, manifests
```

**Ingestion.** A file is parsed in a worker process (text per page or slide, with
headings prepended), split into overlapping ~512-character chunks, tables detected
and rendered as markdown, images extracted. A summary chunk is synthesised and the
document number and revision are extracted once by the model. Everything is embedded
in batches and stored in Qdrant with its metadata. Documents are identified by a
hash of their content, so re-ingesting the same file is a no-op.

**Query.** The question is embedded and searched semantically, *and* searched by
keyword with BM25. The two rankings are fused (Reciprocal Rank Fusion), then the top
candidates are rescored by a cross-encoder. Semantic search misses exact tokens —
part numbers, acronyms, numeric units — and BM25 rescues them; that's why it's
hybrid. Selected chunks are then expanded with their table siblings and their page
neighbours, so the model sees whole tables and whole paragraphs rather than
fragments, and the assembled context is passed to the model with a system prompt
that lists the permitted source files explicitly.

**Air-gap enforcement** is in code, not deployment discipline: Haystack telemetry is
force-disabled at import time, the HTTP client is built with `trust_env=False` so a
corporate proxy can't intercept traffic bound for local Ollama, and Qdrant is bound
to loopback with its own telemetry off. `tests/test_airgap.py` pins all of it.

`CLAUDE.md` in the repository has the full architecture notes, the invariants, and
the reasoning behind every bound in the pipeline. Read it before changing anything
in `retriever.py`.
