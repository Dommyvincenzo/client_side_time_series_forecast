<!-- P2M_REPORT -->
<!-- GENERATED at 2025-11-06 10:37:29 -->
# Project Export: client_side_xgboost

## Overview

- Root: `/home/skinner/client_side_xgboost`
- Files: **29**
- Total size: **936701 bytes**
- Total LOC: 1248 | SLOC: 1018 | TODOs: 0

### Language mix
- plain: 12
- markdown: 7
- typescript: 4
- yaml: 2
- dockerfile: 1
- json: 1
- html: 1
- css: 1

### Top 12 largest files (bytes)
- `vite/app/public/vendor/ml-xgboost/xgboost.wasm` — 884774 bytes
- `vite/app/src/main.ts` — 11407 bytes
- `data.xlsx` — 6996 bytes
- `vite/app/public/favicon.ico` — 5153 bytes
- `CODE_OF_CONDUCT.md` — 4085 bytes
- `vite/app/src/xgb.ts` — 3670 bytes
- `data.csv` — 3154 bytes
- `vite/app/src/api.ts` — 3060 bytes
- `vite/app/src/style.css` — 1807 bytes
- `vite/app/public/vite.svg` — 1497 bytes
- `vite/app/src/typescript.svg` — 1435 bytes
- `SECURITY.md` — 1205 bytes

### Top 12 longest files (LOC)
- `vite/app/src/main.ts` — 368 LOC
- `vite/app/src/style.css` — 113 LOC
- `vite/app/src/xgb.ts` — 104 LOC
- `data.csv` — 91 LOC
- `vite/app/src/api.ts` — 82 LOC
- `CODE_OF_CONDUCT.md` — 68 LOC
- `.gitignore` — 59 LOC
- `.github/workflows/pages.yml` — 41 LOC
- `SECURITY.md` — 38 LOC
- `README.md` — 36 LOC
- `.dockerignore` — 34 LOC
- `.github/ISSUE_TEMPLATE/bug_report.md` — 30 LOC

### Project tree (included subset)
```
client_side_xgboost/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── bug_report.md
│   ├── workflows/
│   │   └── pages.yml
│   └── pull_request_template.md
├── vite/
│   ├── app/
│   │   ├── public/
│   │   │   ├── vendor/
│   │   │   │   └── ml-xgboost/
│   │   │   │       └── xgboost.wasm
│   │   │   ├── apple-touch-icon.png
│   │   │   ├── favicon.ico
│   │   │   ├── favicon.svg
│   │   │   └── vite.svg
│   │   ├── src/
│   │   │   ├── api.ts
│   │   │   ├── main.ts
│   │   │   ├── style.css
│   │   │   ├── typescript.svg
│   │   │   └── xgb.ts
│   │   ├── .gitignore
│   │   ├── index.html
│   │   ├── tsconfig.json
│   │   └── vite.config.ts
│   ├── Dockerfile
│   └── README.md
├── .dockerignore
├── .env
├── .gitignore
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── data.csv
├── data.xlsx
├── docker-compose.yml
├── README.md
└── SECURITY.md
```

## Table of contents (files)

- 1. [.dockerignore](#.dockerignore)
- 2. [.env](#.env)
- 3. [.github/ISSUE_TEMPLATE/bug_report.md](#.github-ISSUE_TEMPLATE-bug_report.md)
- 4. [.github/pull_request_template.md](#.github-pull_request_template.md)
- 5. [.github/workflows/pages.yml](#.github-workflows-pages.yml)
- 6. [.gitignore](#.gitignore)
- 7. [CODE_OF_CONDUCT.md](#CODE_OF_CONDUCT.md)
- 8. [CONTRIBUTING.md](#CONTRIBUTING.md)
- 9. [data.csv](#data.csv)
- 10. [data.xlsx](#data.xlsx)
- 11. [docker-compose.yml](#docker-compose.yml)
- 12. [README.md](#README.md)
- 13. [SECURITY.md](#SECURITY.md)
- 14. [vite/app/.gitignore](#vite-app-.gitignore)
- 15. [vite/app/index.html](#vite-app-index.html)
- 16. [vite/app/public/apple-touch-icon.png](#vite-app-public-apple-touch-icon.png)
- 17. [vite/app/public/favicon.ico](#vite-app-public-favicon.ico)
- 18. [vite/app/public/favicon.svg](#vite-app-public-favicon.svg)
- 19. [vite/app/public/vendor/ml-xgboost/xgboost.wasm](#vite-app-public-vendor-ml-xgboost-xgboost.wasm)
- 20. [vite/app/public/vite.svg](#vite-app-public-vite.svg)
- 21. [vite/app/src/api.ts](#vite-app-src-api.ts)
- 22. [vite/app/src/main.ts](#vite-app-src-main.ts)
- 23. [vite/app/src/style.css](#vite-app-src-style.css)
- 24. [vite/app/src/typescript.svg](#vite-app-src-typescript.svg)
- 25. [vite/app/src/xgb.ts](#vite-app-src-xgb.ts)
- 26. [vite/app/tsconfig.json](#vite-app-tsconfig.json)
- 27. [vite/app/vite.config.ts](#vite-app-vite.config.ts)
- 28. [vite/Dockerfile](#vite-Dockerfile)
- 29. [vite/README.md](#vite-README.md)

---

## Files

<a id=".dockerignore"></a>
### 1. `.dockerignore`
- Size: 305 bytes | LOC: 34 | SLOC: 29 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 4ddfad028bab

#### Brief
# Git
.git

#### Auto Summary
# Git

#### Content

```
# Git
.git
.gitignore

# Python cache
__pycache__/
*.py[cod]
*.pyo
*.pyd

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Local env/config
.env
*.env
*.local

# OS / IDE
.DS_Store
Thumbs.db
desktop.ini
.vscode/
.idea/

# Build artifacts
dist/
build/
web-build/
.expo/
.expo-shared/
```

<a id=".env"></a>
### 2. `.env`
- Size: 597 bytes | LOC: 26 | SLOC: 22 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: a9e6437daaf5

#### Brief
# Postgres
POSTGRES_USER=postgres

#### Auto Summary
# Postgres

#### Content

```
# Postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres

# Backend
BACKEND_PORT=8000

# Auth
AUTH_SECRET=change-this-to-a-long-random-string
AUTH_EXPIRE_MINUTES=60

# DB
DB_HOST=db
DB_PORT=5432
DB_NAME=app_db
DB_USER=app_user
DB_PASS=app_pass
DATABASE_URL=postgresql+psycopg[binary]://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}

# Frontend (Expo)
EXPO_TUNNEL=false
EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0
EXPO_PUBLIC_API_HOST=${REACT_NATIVE_PACKAGER_HOSTNAME}
EXPO_PUBLIC_API_PORT=8000
EXPO_PUBLIC_API_BASE=http://${EXPO_PUBLIC_API_HOST}:${EXPO_PUBLIC_API_PORT}
```

<a id=".github-ISSUE_TEMPLATE-bug_report.md"></a>
### 3. `.github/ISSUE_TEMPLATE/bug_report.md`
- Size: 666 bytes | LOC: 30 | SLOC: 24 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 4b4d2b96744a

#### Brief
---
name: Bug Report

#### Auto Summary
Description

#### Content (verbatim)

```markdown
---
name: Bug Report
about: Create a report to help us improve
title: "[Bug] "
labels: bug
assignees: ''
---

## Description
<!-- A clear and concise description of what the bug is -->

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
<!-- A clear and concise description of what you expected to happen -->

## Screenshots
<!-- If applicable, add screenshots to help explain your problem -->

## Environment
- OS: [e.g. Ubuntu 22.04]
- Browser/Version: [e.g. Chrome 117]
- Node/Python Version: [e.g. Node 18, Python 3.10]

## Additional Context
<!-- Add any other context about the problem here -->
```

<a id=".github-pull_request_template.md"></a>
### 4. `.github/pull_request_template.md`
- Size: 744 bytes | LOC: 27 | SLOC: 21 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 1b6e1aab4d9a

#### Brief
# Pull Request

#### Auto Summary
Pull Request

#### Content (verbatim)

```markdown
# Pull Request

## Overview
<!-- Briefly describe the purpose of this PR and the problem it solves   -->

## Changes
<!-- List the main changes made in this PR -->
- 

## Testing
<!-- Describe how you tested the changes -->
- [ ] Built and ran locally without errors
- [ ] All tests passed
- [ ] Verified functionality manually (describe how)

## Related Issues
<!-- Link to related issues if applicable -->
- Closes #

## Checklist
- [ ] Code is clean and free of unnecessary comments/debug prints
- [ ] Proper naming conventions and documentation are followed
- [ ] Updated documentation/README if necessary
- [ ] CI pipeline passes successfully

## Notes for Reviewers
<!-- Additional context or things reviewers should pay attention to -->
```

<a id=".github-workflows-pages.yml"></a>
### 5. `.github/workflows/pages.yml`
- Size: 917 bytes | LOC: 41 | SLOC: 41 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: a65c0fba07f2

#### Brief
name: pages
on:

#### Auto Summary
name: pages

#### Content

```yaml
name: pages
on:
  push:
    branches: [ main ]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: "pages"
  cancel-in-progress: true
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20
      - name: Install deps
        working-directory: ./vite/app
        run: npm i
      - name: Build
        working-directory: ./vite/app
        run: npm run build:demo
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./vite/app/dist
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

<a id=".gitignore"></a>
### 6. `.gitignore`
- Size: 580 bytes | LOC: 59 | SLOC: 50 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 1c9cbc30ebd8

#### Brief
# Python
__pycache__/

#### Auto Summary
# Python

#### Content

```
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg
*.egg-info/
.eggs/
*.log
*.sqlite3
db.sqlite3

# Virtual env
.venv/
env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints/

# Node / React Native (Expo)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.expo/
.expo-shared/
web-build/
dist/
build/

# macOS / Linux / Windows
.DS_Store
Thumbs.db
desktop.ini

# IDE
.vscode/
.idea/
*.swp

# Docker
*.pid
*.sock
*.tar

# coverage reports (backend & frontend)
coverage/
backend/app/coverage/
frontend/app/coverage/

# Jest cache
frontend/app/.jest-cache/

# mat
mat/
vite_*.md
```

<a id="CODE_OF_CONDUCT.md"></a>
### 7. `CODE_OF_CONDUCT.md`
- Size: 4085 bytes | LOC: 68 | SLOC: 50 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 724cc33f0dca

#### Brief
# Contributor Covenant Code of Conduct

#### Auto Summary
Contributor Covenant Code of Conduct

#### Content (verbatim)

```markdown
# Contributor Covenant Code of Conduct

## Our Pledge
We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

## Our Standards
Examples of behavior that contributes to a positive environment include:
- Demonstrating empathy and kindness toward other people
- Being respectful of differing opinions, viewpoints, and experiences
- Giving and gracefully accepting constructive feedback
- Taking responsibility and apologizing to those affected by our mistakes
- Focusing on what is best for the community, not just for ourselves

Examples of unacceptable behavior include:
- The use of sexualized language or imagery, and sexual attention or advances
- Trolling, insulting or derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others’ private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Enforcement Responsibilities
Community leaders are responsible for clarifying and enforcing our standards of acceptable behavior and will take appropriate and fair corrective action in response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

## Scope
This Code of Conduct applies within all community spaces, and also applies when an individual is officially representing the community in public spaces.

## Enforcement
Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the maintainers of this project. All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the reporter of any incident.

## Enforcement Guidelines
Community leaders will follow these guidelines in determining the consequences for any action they deem in violation of this Code of Conduct:

1. **Correction**  
- *Impact*: Use of inappropriate language or other behavior deemed unprofessional.  
- *Consequence*: A private, written warning, with clarity around the nature of the violation and an explanation of why the behavior was inappropriate.

2. **Warning**  
- *Impact*: A violation through a single incident or series of actions.  
- *Consequence*: A warning with consequences for continued behavior. No interaction with the people involved, including unsolicited interaction with those enforcing the Code of Conduct, for a specified period of time.

3. **Temporary Ban**  
- *Impact*: A serious violation of community standards, including sustained inappropriate behavior.  
- *Consequence*: A temporary ban from any sort of interaction or public communication with the community for a specified period of time.

4. **Permanent Ban**  
- *Impact*: Demonstrating a pattern of violation of community standards, including sustained inappropriate behavior, harassment of an individual, or aggression toward or disparagement of classes of individuals.  
- *Consequence*: A permanent ban from any sort of public interaction within the
   community.

## Attribution

This Code of Conduct is adapted from the  
- [Contributor Covenant][v2.1], version 2.1, available at [https://www.contributor-covenant.org/version/2/1/code_of_conduct.html][v2.1].

Community Impact Guidelines were inspired by  
- [Mozilla’s code of conduct enforcement ladder][mozilla-coc].

For answers to common questions about this code of conduct, see the FAQ at  
- [Contributor Covenant FAQ][faq].

[v2.1]: https://www.contributor-covenant.org/version/2/1/code_of_conduct.html
[mozilla-coc]: https://github.com/mozilla/diversity
[faq]: https://www.contributor-covenant.org/faq
```

<a id="CONTRIBUTING.md"></a>
### 8. `CONTRIBUTING.md`
- Size: 834 bytes | LOC: 25 | SLOC: 19 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 12ea21d94090

#### Brief
# Contributing Guidelines

#### Auto Summary
Contributing Guidelines

#### Content (verbatim)

```markdown
# Contributing Guidelines

Thank you for considering contributing to this project!  
We welcome bug reports, feature requests, and pull requests.  
Please follow the guidelines below to make the process smooth for everyone.

---

## How to Contribute

### 1. Reporting Issues
- Check the [issue tracker](../../issues) to avoid duplicates.
- Use the appropriate [issue template](.github/ISSUE_TEMPLATE/) when creating a new issue.
- Provide as much detail as possible (steps to reproduce, expected behavior, environment, etc.).

### 2. Suggesting Features
- Open a new **Feature Request** issue.
- Explain the problem your idea solves.
- Provide examples, use cases, or references if possible.

### 3. Submitting Pull Requests
Fork the repository and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
```

<a id="data.csv"></a>
### 9. `data.csv`
- Size: 3154 bytes | LOC: 91 | SLOC: 91 | TODOs: 0 | Modified: 2025-11-06 10:29:59 | SHA1: 3f88add2462d

#### Brief
datetime,item_a,item_b,item_c
2025-01-01 00:00:00+09:00,90,80,67

#### Auto Summary
datetime,item_a,item_b,item_c

#### Content

```
datetime,item_a,item_b,item_c
2025-01-01 00:00:00+09:00,90,80,67
2025-01-02 00:00:00+09:00,61,80,10
2025-01-03 00:00:00+09:00,82,88,82
2025-01-04 00:00:00+09:00,1,39,73
2025-01-05 00:00:00+09:00,50,98,57
2025-01-06 00:00:00+09:00,18,78,68
2025-01-07 00:00:00+09:00,76,38,79
2025-01-08 00:00:00+09:00,85,69,79
2025-01-09 00:00:00+09:00,18,35,18
2025-01-10 00:00:00+09:00,7,40,2
2025-01-11 00:00:00+09:00,43,22,35
2025-01-12 00:00:00+09:00,20,62,26
2025-01-13 00:00:00+09:00,48,44,62
2025-01-14 00:00:00+09:00,63,10,79
2025-01-15 00:00:00+09:00,9,54,50
2025-01-16 00:00:00+09:00,9,43,1
2025-01-17 00:00:00+09:00,67,61,67
2025-01-18 00:00:00+09:00,15,10,33
2025-01-19 00:00:00+09:00,10,42,13
2025-01-20 00:00:00+09:00,18,15,54
2025-01-21 00:00:00+09:00,95,72,16
2025-01-22 00:00:00+09:00,13,65,20
2025-01-23 00:00:00+09:00,81,71,40
2025-01-24 00:00:00+09:00,98,67,6
2025-01-25 00:00:00+09:00,45,56,41
2025-01-26 00:00:00+09:00,76,40,23
2025-01-27 00:00:00+09:00,39,52,32
2025-01-28 00:00:00+09:00,25,97,42
2025-01-29 00:00:00+09:00,10,59,19
2025-01-30 00:00:00+09:00,49,74,30
2025-01-31 00:00:00+09:00,97,95,70
2025-02-01 00:00:00+09:00,21,2,83
2025-02-02 00:00:00+09:00,28,58,86
2025-02-03 00:00:00+09:00,36,11,11
2025-02-04 00:00:00+09:00,25,7,55
2025-02-05 00:00:00+09:00,91,25,16
2025-02-06 00:00:00+09:00,33,46,28
2025-02-07 00:00:00+09:00,6,53,2
2025-02-08 00:00:00+09:00,65,12,16
2025-02-09 00:00:00+09:00,20,35,70
2025-02-10 00:00:00+09:00,67,86,41
2025-02-11 00:00:00+09:00,26,55,29
2025-02-12 00:00:00+09:00,70,29,65
2025-02-13 00:00:00+09:00,95,52,34
2025-02-14 00:00:00+09:00,99,57,33
2025-02-15 00:00:00+09:00,78,92,35
2025-02-16 00:00:00+09:00,44,86,90
2025-02-17 00:00:00+09:00,10,65,8
2025-02-18 00:00:00+09:00,66,40,62
2025-02-19 00:00:00+09:00,88,81,7
2025-02-20 00:00:00+09:00,8,54,94
2025-02-21 00:00:00+09:00,38,31,38
2025-02-22 00:00:00+09:00,7,59,72
2025-02-23 00:00:00+09:00,32,55,47
2025-02-24 00:00:00+09:00,56,88,1
2025-02-25 00:00:00+09:00,7,96,35
2025-02-26 00:00:00+09:00,11,20,83
2025-02-27 00:00:00+09:00,37,72,11
2025-02-28 00:00:00+09:00,76,33,80
2025-03-01 00:00:00+09:00,97,19,66
2025-03-02 00:00:00+09:00,42,89,82
2025-03-03 00:00:00+09:00,38,51,45
2025-03-04 00:00:00+09:00,36,25,64
2025-03-05 00:00:00+09:00,37,12,87
2025-03-06 00:00:00+09:00,38,37,80
2025-03-07 00:00:00+09:00,38,38,72
2025-03-08 00:00:00+09:00,13,87,50
2025-03-09 00:00:00+09:00,4,66,18
2025-03-10 00:00:00+09:00,60,77,33
2025-03-11 00:00:00+09:00,35,26,23
2025-03-12 00:00:00+09:00,25,48,22
2025-03-13 00:00:00+09:00,12,66,44
2025-03-14 00:00:00+09:00,69,44,35
2025-03-15 00:00:00+09:00,8,32,29
2025-03-16 00:00:00+09:00,3,59,53
2025-03-17 00:00:00+09:00,27,64,93
2025-03-18 00:00:00+09:00,97,68,8
2025-03-19 00:00:00+09:00,40,47,40
2025-03-20 00:00:00+09:00,79,2,2
2025-03-21 00:00:00+09:00,71,55,17
2025-03-22 00:00:00+09:00,11,78,39
2025-03-23 00:00:00+09:00,20,96,43
2025-03-24 00:00:00+09:00,94,60,30
2025-03-25 00:00:00+09:00,82,77,73
2025-03-26 00:00:00+09:00,35,11,20
2025-03-27 00:00:00+09:00,23,50,61
2025-03-28 00:00:00+09:00,95,14,35
2025-03-29 00:00:00+09:00,43,42,78
2025-03-30 00:00:00+09:00,8,2,1
2025-03-31 00:00:00+09:00,51,82,54
```

<a id="data.xlsx"></a>
### 10. `data.xlsx`
- Size: 6996 bytes | LOC: 0 | SLOC: 0 | TODOs: 0 | Modified: 2025-11-06 10:31:20 | SHA1: 
#### Content

```
```

<a id="docker-compose.yml"></a>
### 11. `docker-compose.yml`
- Size: 385 bytes | LOC: 18 | SLOC: 18 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: ceb75e0b3378

#### Brief
services:
  vite:

#### Auto Summary
services:

#### Content

```yaml
services:
  vite:
    build:
      context: ./vite
      dockerfile: Dockerfile
    container_name: vite
    working_dir: /app
    stdin_open: true
    tty: true
    volumes:
      - ./vite/app:/app
      - /app/node_modules
    ports:
      - "5173:5173"  # Vite
    restart: unless-stopped
    command: ["npm","run","dev","--","--host","0.0.0.0","--port","5173"]
volumes:
  db_data:
```

<a id="README.md"></a>
### 12. `README.md`
- Size: 764 bytes | LOC: 36 | SLOC: 23 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: a50c3ebe05a7

#### Brief
# [Client Side XGBoost](https://github.com/europanite/client_side_xgboost "Client Side XGBoost")

#### Auto Summary
[Client Side XGBoost](https://github.com/europanite/client_side_xgboost "Client Side XGBoost")

#### Content (verbatim)

```markdown
# [Client Side XGBoost](https://github.com/europanite/client_side_xgboost "Client Side XGBoost")


**full-stack development environment** using:
- **Vite**: [Vite](https://vite.dev/) 
- **Container**: [Docker Compose](https://docs.docker.com/compose/) for consistent development setup

---

## 📦 Services

- **db**: PostgreSQL  
  - Port: `5432`  

---

## 🚀 Getting Started

### 1. Prerequisites
- [Docker Compose](https://docs.docker.com/compose/)
- [Expo](https://expo.dev/)　and [Expo Go](https://expo.dev/go) (for Android/iOS testing)　

### 2. Build and start all services:

```bash
# set environment variables:
export REACT_NATIVE_PACKAGER_HOSTNAME=${YOUR_HOST}

# Build the image
docker compose build

# Run the container
docker compose up
```

---
```

<a id="SECURITY.md"></a>
### 13. `SECURITY.md`
- Size: 1205 bytes | LOC: 38 | SLOC: 24 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 2dedcdd6db11

#### Brief
# Security Policy

#### Auto Summary
Security Policy

#### Content (verbatim)

```markdown
# Security Policy

## Supported Versions

The following table shows which versions of `standard_react_fastapi_environment` are currently being supported with security updates.

| Version | Supported          |
|---------|--------------------|
| main    | :white_check_mark: |

We only provide security updates and fixes for the latest code on the **main** branch.

---

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please help us keep the community safe by following these steps:

- Provide as much detail as possible:
   - A clear description of the vulnerability
   - Steps to reproduce the issue
   - The potential impact
   - Any suggested fixes or mitigations

---

## Security Best Practices for Users

- Always pull the latest image or rebuild the environment to ensure patched dependencies.
- Avoid exposing ports publicly unless necessary.
- Use strong passwords and secrets when connecting to external resources.
- Regularly update Python packages and system dependencies.

---

## Acknowledgements

We deeply appreciate the efforts of security researchers and contributors who help us improve the security of `standard_react_fastapi_environment`.
```

<a id="vite-app-.gitignore"></a>
### 14. `vite/app/.gitignore`
- Size: 253 bytes | LOC: 24 | SLOC: 22 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: f97396057473

#### Brief
# Logs
logs

#### Auto Summary
# Logs

#### Content

```
# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

node_modules
dist
dist-ssr
*.local

# Editor directories and files
.vscode/*
!.vscode/extensions.json
.idea
.DS_Store
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?
```

<a id="vite-app-index.html"></a>
### 15. `vite/app/index.html`
- Size: 199 bytes | LOC: 8 | SLOC: 8 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 3fb88df42d18

#### Brief
<!doctype html>
<head>

#### Auto Summary
<!doctype html>

#### Content

```html
<!doctype html>
<head>
  <meta charset="utf-8" />
  <title>Client Side XGBoost</title>
</head>
<body style="margin:0;background:#000000">
  <script type="module" src="/src/main.ts"></script>
</body>
```

<a id="vite-app-public-apple-touch-icon.png"></a>
### 16. `vite/app/public/apple-touch-icon.png`
- Size: 626 bytes | LOC: 0 | SLOC: 0 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 
#### Content

```
```

<a id="vite-app-public-favicon.ico"></a>
### 17. `vite/app/public/favicon.ico`
- Size: 5153 bytes | LOC: 0 | SLOC: 0 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 
#### Content

```
```

<a id="vite-app-public-favicon.svg"></a>
### 18. `vite/app/public/favicon.svg`
- Size: 430 bytes | LOC: 8 | SLOC: 8 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: 5a91e304d7df

#### Brief
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="61" fill="#4F46E5"/>

#### Auto Summary
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">

#### Content

```
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="61" fill="#4F46E5"/>
  <!-- 'E' monogram -->
  <rect x="92" y="92" width="56" height="328" fill="#FFFFFF"/>
  <rect x="92" y="92" width="328" height="56" fill="#FFFFFF"/>
  <rect x="92" y="228" width="284" height="56" fill="#FFFFFF"/>
  <rect x="92" y="364" width="328" height="56" fill="#FFFFFF"/>
</svg>
```

<a id="vite-app-public-vendor-ml-xgboost-xgboost.wasm"></a>
### 19. `vite/app/public/vendor/ml-xgboost/xgboost.wasm`
- Size: 884774 bytes | LOC: 0 | SLOC: 0 | TODOs: 0 | Modified: 2025-11-06 10:01:50 | SHA1: 
#### Content

```
```

<a id="vite-app-public-vite.svg"></a>
### 20. `vite/app/public/vite.svg`
- Size: 1497 bytes | LOC: 1 | SLOC: 1 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: f7f39d7237b7

#### Brief
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="31.88" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 257"><defs><linearGradient id="IconifyId1813088fe1fbc01fb466" x1="-.828%" x2="57.636%" y1="7.652%" y2="78.411%"><stop offset="0%" stop-color="#41D1FF"></stop><stop offset="100%" stop-color="#BD34FE"></stop></linearGradient><linearGradient id="IconifyId1813088fe1fbc01fb467" x1="43.376%" x2="50.316%" y1="2.242%" y2="89.03%"><stop offset="0%" stop-color="#FFEA83"></stop><stop offset="8.333%" stop-color="#FFDD35"></stop><stop offset="100%" stop-color="#FFA800"></stop></linearGradient></defs><path fill="url(#IconifyId1813088fe1fbc01fb466)" d="M255.153 37.938L134.897 252.976c-2.483 4.44-8.862 4.466-11.382.048L.875 37.958c-2.746-4.814 1.371-10.646 6.827-9.67l120.385 21.517a6.537 6.537 0 0 0 2.322-.004l117.867-21.483c5.438-.991 9.574 4.796 6.877 9.62Z"></path><path fill="url(#IconifyId1813088fe1fbc01fb467)" d="M185.432.063L96.44 17.501a3.268 3.268 0 0 0-2.634 3.014l-5.474 92.456a3.268 3.268 0 0 0 3.997 3.378l24.777-5.718c2.318-.535 4.413 1.507 3.936 3.838l-7.361 36.047c-.495 2.426 1.782 4.5 4.151 3.78l15.304-4.649c2.372-.72 4.652 1.36 4.15 3.788l-11.698 56.621c-.732 3.542 3.979 5.473 5.943 2.437l1.313-2.028l72.516-144.72c1.215-2.423-.88-5.186-3.54-4.672l-25.505 4.922c-2.396.462-4.435-1.77-3.759-4.114l16.646-57.705c.677-2.35-1.37-4.583-3.769-4.113Z"></path></svg>

#### Auto Summary
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="31.88" height="32" preserveAspectRatio="xMidYMid 

#### Content

```
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="31.88" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 257"><defs><linearGradient id="IconifyId1813088fe1fbc01fb466" x1="-.828%" x2="57.636%" y1="7.652%" y2="78.411%"><stop offset="0%" stop-color="#41D1FF"></stop><stop offset="100%" stop-color="#BD34FE"></stop></linearGradient><linearGradient id="IconifyId1813088fe1fbc01fb467" x1="43.376%" x2="50.316%" y1="2.242%" y2="89.03%"><stop offset="0%" stop-color="#FFEA83"></stop><stop offset="8.333%" stop-color="#FFDD35"></stop><stop offset="100%" stop-color="#FFA800"></stop></linearGradient></defs><path fill="url(#IconifyId1813088fe1fbc01fb466)" d="M255.153 37.938L134.897 252.976c-2.483 4.44-8.862 4.466-11.382.048L.875 37.958c-2.746-4.814 1.371-10.646 6.827-9.67l120.385 21.517a6.537 6.537 0 0 0 2.322-.004l117.867-21.483c5.438-.991 9.574 4.796 6.877 9.62Z"></path><path fill="url(#IconifyId1813088fe1fbc01fb467)" d="M185.432.063L96.44 17.501a3.268 3.268 0 0 0-2.634 3.014l-5.474 92.456a3.268 3.268 0 0 0 3.997 3.378l24.777-5.718c2.318-.535 4.413 1.507 3.936 3.838l-7.361 36.047c-.495 2.426 1.782 4.5 4.151 3.78l15.304-4.649c2.372-.72 4.652 1.36 4.15 3.788l-11.698 56.621c-.732 3.542 3.979 5.473 5.943 2.437l1.313-2.028l72.516-144.72c1.215-2.423-.88-5.186-3.54-4.672l-25.505 4.922c-2.396.462-4.435-1.77-3.759-4.114l16.646-57.705c.677-2.35-1.37-4.583-3.769-4.113Z"></path></svg>
```

<a id="vite-app-src-api.ts"></a>
### 21. `vite/app/src/api.ts`
- Size: 3060 bytes | LOC: 82 | SLOC: 65 | TODOs: 0 | Modified: 2025-11-06 08:46:00 | SHA1: acba8fdb7e0b

#### Brief
Utility helpers for parsing and feature engineering (English only)

#### Auto Summary
// Utility helpers for parsing and feature engineering (English only)

#### Content

```typescript

// Utility helpers for parsing and feature engineering (English only)

/** Parse a CSV string to rows of objects */
export function parseCSV(csvText: string): { rows: any[]; headers: string[] } {
  // PapaParse is heavy to import eagerly; for simplicity, a tiny CSV parser is ok for demo
  // But we will import papaparse dynamically for robustness
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  return window.__parseCSV
    ? window.__parseCSV(csvText)
    : simpleCSV(csvText);
}

/** Minimal CSV parser (fallback) */
function simpleCSV(text: string): { rows: any[]; headers: string[] } {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim());
    const obj: Record<string, any> = {};
    headers.forEach((h, i) => (obj[h] = vals[i]));
    return obj;
  });
  return { rows, headers };
}

/** Build features for tabular tree models: other series + time features */
export function buildFeatures(
  rows: any[],
  datetimeKey: string,
  targetKey: string
): { X: number[][]; y: number[]; lastFeatureRow: number[] } {
  const headers = Object.keys(rows[0] ?? {});
  const featureKeys = headers.filter(
    (h) => h !== datetimeKey && h !== targetKey
  );
  // time index engineered features (sin/cos seasonality + index)
  const X: number[][] = [];
  const y: number[] = [];
  const toNum = (v: any) => (v === "" || v == null ? NaN : Number(v));

  rows.forEach((r, idx) => {
    const t = idx;
    const feats: number[] = [];
    // other series
    featureKeys.forEach((k) => feats.push(toNum(r[k])));
    // time encodings
    feats.push(t);
    feats.push(Math.sin((2 * Math.PI * t) / 24)); // daily-like
    feats.push(Math.cos((2 * Math.PI * t) / 24));
    feats.push(Math.sin((2 * Math.PI * t) / 168)); // weekly-like
    feats.push(Math.cos((2 * Math.PI * t) / 168));
    X.push(feats);
    y.push(toNum(r[targetKey]));
  });

  // next-step features = last row with t = rows.length
  const last = rows[rows.length - 1];
  const tNext = rows.length;
  const nextFeats: number[] = [];
  featureKeys.forEach((k) => nextFeats.push(toNum(last[k])));
  nextFeats.push(tNext);
  nextFeats.push(Math.sin((2 * Math.PI * tNext) / 24));
  nextFeats.push(Math.cos((2 * Math.PI * tNext) / 24));
  nextFeats.push(Math.sin((2 * Math.PI * tNext) / 168));
  nextFeats.push(Math.cos((2 * Math.PI * tNext) / 168));

  return { X, y, lastFeatureRow: nextFeats };
}

/** Safely parse XLSX ArrayBuffer into rows */
export function parseXLSX(buf: ArrayBuffer): { rows: any[]; headers: string[] } {
  const XLSX = (window as any).XLSX;
  if (!XLSX) throw new Error("XLSX library not found on window");
  const wb = XLSX.read(new Uint8Array(buf), { type: "array" });
  const wsName = wb.SheetNames[0];
  const ws = wb.Sheets[wsName];
  const rows: any[] = XLSX.utils.sheet_to_json(ws, { raw: true });
  const headers = rows.length ? Object.keys(rows[0]) : [];
  return { rows, headers };
}
```

<a id="vite-app-src-main.ts"></a>
### 22. `vite/app/src/main.ts`
- Size: 11407 bytes | LOC: 368 | SLOC: 283 | TODOs: 0 | Modified: 2025-11-06 10:33:28 | SHA1: dac0bc4a6c0f

#### Brief
Time-series forecaster demo (CSV/XLSX import → plot → pick target → train XGBoost → predict +1)
Clean rewrite: rely on xgb.ts for WASM locateFile & factory normalization only.

#### Auto Summary
// Time-series forecaster demo (CSV/XLSX import → plot → pick target → train XGBoost → predict +1)

#### Content

```typescript
// Time-series forecaster demo (CSV/XLSX import → plot → pick target → train XGBoost → predict +1)
// Clean rewrite: rely on xgb.ts for WASM locateFile & factory normalization only.

import "./style.css";
import { parseCSV, parseXLSX, buildFeatures } from "./api";
import { initXGBoostCtor } from "./xgb";

let Chart: any;

type TrainState = {
  targetKey: string | null;
  datetimeKey: string | null;
  rows: any[];
  headers: string[];
  model: any | null;   // Booster instance
  xgbCtor: any | null; // XGBoost constructor/class returned by factory
};

const state: TrainState = {
  targetKey: null,
  datetimeKey: null,
  rows: [],
  headers: [],
  model: null,
  xgbCtor: null,
};

// ---- UI scaffold ------------------------------------------------------------

const app = document.createElement("div");
app.id = "app";
document.body.appendChild(app);

app.innerHTML = `
  <h1>Time-series Forecaster</h1>

  <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center;">
    <input id="file" type="file" accept=".csv, .xlsx" />
    <span id="status" style="opacity:0.8">Status: idle</span>
  </div>

  <div style="margin-top:12px; display:flex; gap:12px; align-items:center;">
    <label>Datetime column:</label>
    <select id="datetimeSelect"></select>

    <label>Target:</label>
    <select id="targetSelect"></select>

    <button id="trainBtn" disabled>Train (XGBoost)</button>
    <button id="predictBtn" disabled>Predict +1</button>
  </div>

  <div id="chartWrap" style="max-width:1100px; width:95vw; height:420px; margin-top:16px;">
    <canvas id="chart"></canvas>
  </div>

  <pre id="out" style="margin-top:16px; padding:12px; background:#0b1020; color:#e6f1ff; border-radius:8px; white-space:pre-wrap;"></pre>
`;

// ---- Element refs -----------------------------------------------------------

const els = {
  file: document.getElementById("file") as HTMLInputElement,
  status: document.getElementById("status") as HTMLSpanElement,
  dtSel: document.getElementById("datetimeSelect") as HTMLSelectElement,
  targetSel: document.getElementById("targetSelect") as HTMLSelectElement,
  train: document.getElementById("trainBtn") as HTMLButtonElement,
  predict: document.getElementById("predictBtn") as HTMLButtonElement,
  chart: document.getElementById("chart") as HTMLCanvasElement,
  out: document.getElementById("out") as HTMLPreElement,
};

function setStatus(s: string) {
  els.status.textContent = `Status: ${s}`;
}

// ---- Chart rendering --------------------------------------------------------

let chartInstance: any = null;

async function ensureChart() {
  if (!Chart) {
    Chart = (await import("chart.js/auto")).default;
  }
}

function fillSelect(sel: HTMLSelectElement, options: string[]) {
  sel.innerHTML = options.map((o) => `<option value="${o}">${o}</option>`).join("");
}

async function renderChart(rows: any[], headers: string[], datetimeKey: string | null) {
  await ensureChart();
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }
  if (!rows?.length || !headers?.length) return;

  // X 軸（日時 or インデックス）
  const xLabels = (datetimeKey && headers.includes(datetimeKey))
    ? rows.map((r) => String(r[datetimeKey] ?? ""))
    : rows.map((_, i) => i);

  // 数値列の判定（先頭～10件でフィニットかを見る）
  const isNumericCol = (key: string) => {
    const n = Math.min(rows.length, 10);
    for (let i = 0; i < n; i++) {
      const v = rows[i]?.[key];
      const num = typeof v === "number" ? v : Number((v ?? "").toString().trim());
      if (Number.isFinite(num)) return true;
    }
    return false;
  };

  // 日時列を除いた “数値として扱える列”
  const seriesKeys = headers.filter((h) => h !== datetimeKey).filter(isNumericCol);
  if (!seriesKeys.length) {
    console.warn("[chart] no numeric series detected");
    return;
  }

  // 数値化（非数は null にしてギャップ扱い）
  const toNumOrNull = (v: any) => {
    const num = typeof v === "number" ? v : Number((v ?? "").toString().trim());
    return Number.isFinite(num) ? num : null;
  };

  const palette = [
    "#22c55e", "#60a5fa", "#f59e0b", "#ef4444",
    "#a78bfa", "#14b8a6", "#e11d48", "#94a3b8",
  ];

  const datasets = seriesKeys.map((k, i) => ({
    label: k,
    data: rows.map((r) => toNumOrNull(r[k])),
    borderColor: palette[i % palette.length],
    backgroundColor: palette[i % palette.length],
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.25,
    spanGaps: true,               // 欠損は飛ばして線をつなぐ
  }));

  // y 軸レンジをデータから自動算出（余白 5%）
  let yMin = Number.POSITIVE_INFINITY;
  let yMax = Number.NEGATIVE_INFINITY;
  for (const ds of datasets) {
    for (const v of ds.data as Array<number | null>) {
      if (v == null) continue;
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    console.warn("[chart] all points are null; nothing to draw");
    return;
  }
  if (yMin === yMax) {
    // 全値同一でも線が見えるように幅を持たせる
    yMin -= 1;
    yMax += 1;
  }
  const pad = (yMax - yMin) * 0.05;
  const suggestedMin = yMin - pad;
  const suggestedMax = yMax + pad;

  // キャンバス実寸の確保（高さ0対策）
  const wrap = document.getElementById("chartWrap") as HTMLDivElement | null;
  if (wrap) {
    wrap.style.minHeight = "320px";
    wrap.style.minWidth = "320px";
  }

  const ctx = els.chart.getContext("2d");
  if (!ctx) return;

  chartInstance = new Chart(ctx, {
    type: "line",
    data: { labels: xLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,                       // 数値配列をそのまま使う
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: true, position: "top" } },
      scales: {
        x: { display: true, grid: { display: false } },
        y: {
          type: "linear",
          display: true,
          grid: { color: "rgba(255,255,255,0.1)" },
          suggestedMin,
          suggestedMax,
        },
      },
    },
  });

  chartInstance.update();
}

function updateSelectors(headers: string[]) {
  fillSelect(els.dtSel, headers);
  fillSelect(els.targetSel, headers);
  const dtCandidate = headers.find(
    (h) => h.toLowerCase().includes("date") || h.toLowerCase().includes("time")
  );
  if (dtCandidate) els.dtSel.value = dtCandidate;
  const targetCandidate = headers.find((h) => h !== els.dtSel.value);
  if (targetCandidate) els.targetSel.value = targetCandidate;
}

// ---- File loading (CSV/XLSX) -----------------------------------------------

async function loadCSVText(text: string) {
  // Prefer PapaParse if available; fall back to tiny parser inside api.ts
  try {
    const Papa = (await import("papaparse")).default;
    (window as any).__parseCSV = (t: string) => {
      const res = Papa.parse(t, { header: true, dynamicTyping: true, skipEmptyLines: true });
      return { rows: res.data as any[], headers: res.meta.fields ?? [] };
    };
  } catch {
    // fallback handled by parseCSV
  }
  const { rows, headers } = parseCSV(text);
  state.rows = rows;
  state.headers = headers;
  updateSelectors(headers);
  state.datetimeKey = els.dtSel.value || null;
  state.targetKey = els.targetSel.value || null;

  await renderChart(rows, headers, state.datetimeKey);
  els.train.disabled = !state.targetKey;
  els.predict.disabled = true;
  setStatus("data loaded");
}

async function loadXLSXBuffer(buf: ArrayBuffer) {
  const XLSX = await import("xlsx");
  (window as any).XLSX = XLSX; // api.ts expects XLSX on window
  const { rows, headers } = parseXLSX(buf);
  state.rows = rows;
  state.headers = headers;
  updateSelectors(headers);
  state.datetimeKey = els.dtSel.value || null;
  state.targetKey = els.targetSel.value || null;

  await renderChart(rows, headers, state.datetimeKey);
  els.train.disabled = !state.targetKey;
  els.predict.disabled = true;
  setStatus("data loaded");
}

// ---- Events ----------------------------------------------------------------

els.file.addEventListener("change", async () => {
  const f = els.file.files?.[0];
  if (!f) return;
  setStatus(`reading ${f.name} ...`);
  const ext = f.name.toLowerCase().split(".").pop();
  try {
    if (ext === "csv") {
      const text = await f.text();
      await loadCSVText(text);
    } else if (ext === "xlsx") {
      const buf = await f.arrayBuffer();
      await loadXLSXBuffer(buf);
    } else {
      throw new Error("Unsupported file type (use .csv or .xlsx)");
    }
  } catch (err: any) {
    setStatus(`error: ${err.message || err}`);
  }
});

els.dtSel.addEventListener("change", async () => {
  state.datetimeKey = els.dtSel.value || null;
  await renderChart(state.rows, state.headers, state.datetimeKey);
});

els.targetSel.addEventListener("change", () => {
  state.targetKey = els.targetSel.value || null;
  els.train.disabled = !state.targetKey;
});

// ---- Train / Predict (ml-xgboost through xgb.ts) ---------------------------

els.train.addEventListener("click", async () => {
  if (!state.rows.length || !state.targetKey) return;

  setStatus("building features ...");
  els.train.disabled = true;
  els.predict.disabled = true;

  const { X, y } = buildFeatures(
    state.rows,
    state.datetimeKey!,
    state.targetKey
  );

  try {
    setStatus("initializing ml-xgboost ...");
    const XGBoost = await initXGBoostCtor();     // ← xgb.ts handles WASM path & export shape
    state.xgbCtor = XGBoost;

    // const booster = new XGBoost({
    //   booster: "gbtree",
    //   objective: "reg:squarederror",
    //   max_depth: 4,
    //   eta: 0.1,
    //   min_child_weight: 1,
    //   subsample: 0.8,
    //   colsample_bytree: 1,
    //   silent: 1,
    //   iterations: 200,
    // });

    const booster = new XGBoost({
      booster: "gbtree",
      // Older XGBoost build in this WASM doesn't support "reg:squarederror"
      // Use the legacy alias:
      objective: "reg:linear",
      max_depth: 4,
      eta: 0.1,
      min_child_weight: 1,
      subsample: 0.8,
      colsample_bytree: 1,
      silent: 1,
      iterations: 200,
      // (optional) eval_metric: "rmse" も指定可
    });

    setStatus("training (ml-xgboost) ...");
    booster.train(X, y);

    state.model = booster;
    setStatus("trained with ml-xgboost");
    els.predict.disabled = false;
  } catch (e: any) {
    console.error(e);
    state.model = null;
    setStatus(`error: failed to initialize or train ml-xgboost (${e?.message || e})`);
  } finally {
    els.train.disabled = false;
  }
});

els.predict.addEventListener("click", () => {
  if (!state.rows.length || !state.targetKey || !state.model) return;
  const { lastFeatureRow } = buildFeatures(
    state.rows,
    state.datetimeKey!,
    state.targetKey
  );
  const pred = state.model.predict([lastFeatureRow]);
  const yhat = Array.isArray(pred) ? Number(pred[0]) : Number(pred);
  els.out.textContent = `Target="${state.targetKey}" → next(+1) forecast: ${
    Number.isFinite(yhat) ? yhat.toFixed(4) : String(yhat)
  }`;
  setStatus("predicted");
});

// Free model if possible
window.addEventListener("beforeunload", () => {
  try { state.model?.free?.(); } catch { /* ignore */ }
});
```

<a id="vite-app-src-style.css"></a>
### 23. `vite/app/src/style.css`
- Size: 1807 bytes | LOC: 113 | SLOC: 99 | TODOs: 0 | Modified: 2025-11-06 09:04:02 | SHA1: f1a33a565712

#### Brief
:root {
  font-family: system-ui, Avenir, Helvetica, Arial, sans-serif;

#### Auto Summary
:root {

#### Content

```css
:root {
  font-family: system-ui, Avenir, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  font-weight: 400;

  color-scheme: light dark;
  color: rgba(255, 255, 255, 0.87);
  background-color: #242424;

  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

a {
  font-weight: 500;
  color: #646cff;
  text-decoration: inherit;
}
a:hover {
  color: #535bf2;
}

body {
  margin: 0;
  display: flex;
  place-items: center;
  min-width: 320px;
  min-height: 100vh;
}

h1 {
  font-size: 3.2em;
  line-height: 1.1;
}

#app {
  max-width: 1280px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}

.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vanilla:hover {
  filter: drop-shadow(0 0 2em #3178c6aa);
}

.card {
  padding: 2em;
}


.read-the-docs {
  color: #888;
}

button {
  border-radius: 8px;
  border: 1px solid transparent;
  padding: 0.6em 1.2em;
  font-size: 1em;
  font-weight: 500;
  font-family: inherit;
  background-color: #1a1a1a;
  cursor: pointer;
  transition: border-color 0.25s;
}
button:hover {
  border-color: #646cff;
}
button:focus,
button:focus-visible {
  outline: 4px auto -webkit-focus-ring-color;
}

@media (prefers-color-scheme: light) {
  :root {
    color: #213547;
    background-color: #ffffff;
  }
  a:hover {
    color: #747bff;
  }
  button {
    background-color: #f9f9f9;
  }
}

/* Chart area sizing (stop vertical overgrowth) */
#chartWrap {
  position: relative;
  height: 420px;        
  max-height: 70vh; 
  margin: 16px auto 0;
  background: #111;
  border-radius: 8px;
}

#chart {
  width: 100% !important;
  height: 100% !important;
  display: block;
}
```

<a id="vite-app-src-typescript.svg"></a>
### 24. `vite/app/src/typescript.svg`
- Size: 1435 bytes | LOC: 1 | SLOC: 1 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: f15a725e7a29

#### Brief
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="32" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 256"><path fill="#007ACC" d="M0 128v128h256V0H0z"></path><path fill="#FFF" d="m56.612 128.85l-.081 10.483h33.32v94.68h23.568v-94.68h33.321v-10.28c0-5.69-.122-10.444-.284-10.566c-.122-.162-20.4-.244-44.983-.203l-44.74.122l-.121 10.443Zm149.955-10.742c6.501 1.625 11.459 4.51 16.01 9.224c2.357 2.52 5.851 7.111 6.136 8.208c.08.325-11.053 7.802-17.798 11.988c-.244.162-1.22-.894-2.317-2.52c-3.291-4.795-6.745-6.867-12.028-7.233c-7.76-.528-12.759 3.535-12.718 10.321c0 1.992.284 3.17 1.097 4.795c1.707 3.536 4.876 5.649 14.832 9.956c18.326 7.883 26.168 13.084 31.045 20.48c5.445 8.249 6.664 21.415 2.966 31.208c-4.063 10.646-14.14 17.879-28.323 20.276c-4.388.772-14.79.65-19.504-.203c-10.28-1.828-20.033-6.908-26.047-13.572c-2.357-2.6-6.949-9.387-6.664-9.874c.122-.163 1.178-.813 2.356-1.504c1.138-.65 5.446-3.129 9.509-5.485l7.355-4.267l1.544 2.276c2.154 3.29 6.867 7.801 9.712 9.305c8.167 4.307 19.383 3.698 24.909-1.26c2.357-2.153 3.332-4.388 3.332-7.68c0-2.966-.366-4.266-1.91-6.501c-1.99-2.845-6.054-5.242-17.595-10.24c-13.206-5.69-18.895-9.224-24.096-14.832c-3.007-3.25-5.852-8.452-7.03-12.8c-.975-3.617-1.22-12.678-.447-16.335c2.723-12.76 12.353-21.659 26.25-24.3c4.51-.853 14.994-.528 19.424.569Z"></path></svg>

#### Auto Summary
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="32" height="32" preserveAspectRatio="xMidYMid mee

#### Content

```
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="32" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 256"><path fill="#007ACC" d="M0 128v128h256V0H0z"></path><path fill="#FFF" d="m56.612 128.85l-.081 10.483h33.32v94.68h23.568v-94.68h33.321v-10.28c0-5.69-.122-10.444-.284-10.566c-.122-.162-20.4-.244-44.983-.203l-44.74.122l-.121 10.443Zm149.955-10.742c6.501 1.625 11.459 4.51 16.01 9.224c2.357 2.52 5.851 7.111 6.136 8.208c.08.325-11.053 7.802-17.798 11.988c-.244.162-1.22-.894-2.317-2.52c-3.291-4.795-6.745-6.867-12.028-7.233c-7.76-.528-12.759 3.535-12.718 10.321c0 1.992.284 3.17 1.097 4.795c1.707 3.536 4.876 5.649 14.832 9.956c18.326 7.883 26.168 13.084 31.045 20.48c5.445 8.249 6.664 21.415 2.966 31.208c-4.063 10.646-14.14 17.879-28.323 20.276c-4.388.772-14.79.65-19.504-.203c-10.28-1.828-20.033-6.908-26.047-13.572c-2.357-2.6-6.949-9.387-6.664-9.874c.122-.163 1.178-.813 2.356-1.504c1.138-.65 5.446-3.129 9.509-5.485l7.355-4.267l1.544 2.276c2.154 3.29 6.867 7.801 9.712 9.305c8.167 4.307 19.383 3.698 24.909-1.26c2.357-2.153 3.332-4.388 3.332-7.68c0-2.966-.366-4.266-1.91-6.501c-1.99-2.845-6.054-5.242-17.595-10.24c-13.206-5.69-18.895-9.224-24.096-14.832c-3.007-3.25-5.852-8.452-7.03-12.8c-.975-3.617-1.22-12.678-.447-16.335c2.723-12.76 12.353-21.659 26.25-24.3c4.51-.853 14.994-.528 19.424.569Z"></path></svg>
```

<a id="vite-app-src-xgb.ts"></a>
### 25. `vite/app/src/xgb.ts`
- Size: 3670 bytes | LOC: 104 | SLOC: 77 | TODOs: 0 | Modified: 2025-11-06 10:11:50 | SHA1: 867e22a36ed2

#### Brief
const WASM_URL = new URL(
  `${import.meta.env.BASE_URL}vendor/ml-xgboost/xgboost.wasm`,

#### Auto Summary
const WASM_URL = new URL(

#### Content

```typescript
const WASM_URL = new URL(
  `${import.meta.env.BASE_URL}vendor/ml-xgboost/xgboost.wasm`,
  self.location.href
).toString();

async function fetchWasmBinary(url: string): Promise<ArrayBuffer> {
  const u = new URL(url);
  u.searchParams.set("t", String(Date.now()));        // bust cache in dev/HMR
  const r = await fetch(u.toString(), { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed to fetch wasm: ${u} status=${r.status} ${r.statusText}`);
  return await r.arrayBuffer();
}

export async function initXGBoostCtor(): Promise<any> {
  // 1) 先に正しい wasm を確保
  const wasmBytes = await fetchWasmBinary(WASM_URL);
  const wasmResp = new Response(new Blob([wasmBytes], { type: "application/wasm" }), {
    status: 200,
    headers: { "Content-Type": "application/wasm" },
  });

  // 2) fetch を一時フック（ml-xgboost の内部 fetch を横取り）
  const origFetch = self.fetch.bind(self);
  (self as any).fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    try {
      const url = typeof input === "string" ? input : (input as Request).url;
      if (url.includes("xgboost.wasm")) {
        // 正しい wasm を常に返す（コピーを返す）
        return Promise.resolve(wasmResp.clone());
      }
    } catch { /* ignore */ }
    return origFetch(input as any, init as any);
  };

  // 3) ここで初めて ml-xgboost を import（プリバンドル済でもOK）
  let mod: any;
  try {
    mod = await import("ml-xgboost");
  } finally {
    // 4) 必ず元に戻す
    (self as any).fetch = origFetch;
  }

  // 5) export 形状を正規化。Emscripten factory に locateFile も渡す（保険）
    const normalize = async (m: any): Promise<any | null> => {
    if (!m) return null;

    // Promise → 展開
    if (typeof m?.then === "function") {
        return await normalize(await m);
    }

    // ESModule default → 展開
    if (m?.default) {
        return await normalize(m.default);
    }

    // 名前付きエクスポートにクラスがいるケース
    if (m?.XGBoost && typeof m.XGBoost === "function") {
        return m.XGBoost; // ← クラスをそのまま返す
    }

    if (typeof m === "function") {
        // クラスかどうかの簡易判定
        const src = Function.prototype.toString.call(m);
        const looksLikeClass =
        src.startsWith("class ") ||
        /class\s+[A-Za-z0-9_]+/.test(src) ||
        ("prototype" in m && Object.getOwnPropertyNames(m.prototype || {}).length > 1);

        if (looksLikeClass) {
        // クラスならそのまま返す（呼び出さない）
        return m;
        }

        // 工場関数の可能性があるので試す。ただしクラスだったら例外で戻る
        try {
        const ctor = await m({
            locateFile: (p: string) => (p.endsWith(".wasm") ? WASM_URL : p),
        });
        return ctor; // ここは"関数（コンストラクタ）"が返る想定
        } catch (e: any) {
        // クラスを factory のように呼んだときの典型的な TypeError
        if (String(e).includes("cannot be invoked without 'new'")) {
            return m; // ← クラスとして扱う
        }
        throw e; // それ以外は本物の異常
        }
    }

    return null;
    };

  const ctor =
    (await normalize(mod)) ??
    (await normalize(mod?.default)) ??
    (await normalize((mod && mod["ml-xgboost"]) || null));

  if (!ctor || typeof ctor !== "function") {
    const keys = Object.keys(mod || {});
    throw new Error(`ml-xgboost export shape unsupported. typeof default=${typeof mod?.default}, keys=${keys.join(",")}`);
  }
  return ctor;
}
```

<a id="vite-app-tsconfig.json"></a>
### 26. `vite/app/tsconfig.json`
- Size: 637 bytes | LOC: 26 | SLOC: 24 | TODOs: 0 | Modified: 2025-11-06 01:22:30 | SHA1: af1f75a8015c

#### Brief
{
  "compilerOptions": {

#### Auto Summary
{

#### Content

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "types": ["vite/client"],
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "verbatimModuleSyntax": true,
    "moduleDetection": "force",
    "noEmit": true,

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "erasableSyntaxOnly": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  },
  "include": ["src"]
}
```

<a id="vite-app-vite.config.ts"></a>
### 27. `vite/app/vite.config.ts`
- Size: 439 bytes | LOC: 14 | SLOC: 13 | TODOs: 0 | Modified: 2025-11-06 09:44:38 | SHA1: c1f0a83b0bae

#### Brief
import { defineConfig } from "vite";

#### Auto Summary
import { defineConfig } from "vite";

#### Content

```typescript
import { defineConfig } from "vite";

export default defineConfig({
  base: "/client_side_xgboost/",           // ← GitHub Pages 用
  assetsInclude: ["**/*.wasm"],            // ← wasmをアセットとして扱う
  optimizeDeps: {
    include: ["ml-xgboost"],               // ← 依存前処理の安定化
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    commonjsOptions: { transformMixedEsModules: true },
  },
});
```

<a id="vite-Dockerfile"></a>
### 28. `vite/Dockerfile`
- Size: 82 bytes | LOC: 6 | SLOC: 5 | TODOs: 0 | Modified: 2025-11-06 08:18:31 | SHA1: c5038ca61dc6

#### Brief
FROM node:20-bullseye

#### Auto Summary
FROM node:20-bullseye

#### Content

```dockerfile
FROM node:20-bullseye

WORKDIR /app
COPY app/package*.json ./
RUN npm ci
COPY . .
```

<a id="vite-README.md"></a>
### 29. `vite/README.md`
- Size: 0 bytes | LOC: 0 | SLOC: 0 | TODOs: 0 | Modified: 2025-11-06 01:14:52 | SHA1: 
#### Content (verbatim)

```markdown

```
