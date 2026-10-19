# Inference Engines: From First Principles to Serving Systems

A research textbook covering model execution, hardware cost models, state management, inference kernels, serving systems, quantization, speculative decoding and an engine build plan.

The HTML edition contains 15 technical chapters, a study guide, glossary, exercises with worked answers, references and source provenance. Its native MathML equations, CSS and reader controls are embedded, so the book reads offline without a CDN.

Research cutoff: **19 October 2026**. This is a selective literature review; published benchmark results have not been independently reproduced.

## Read locally

Open [the textbook](research/textbook/index.html) in a modern browser. See [reader and rebuild instructions](research/textbook/README.md).

To regenerate after editing the reviewed reports or teaching supplements:

```powershell
python research/textbook/assemble.py
python research/textbook/build.py
python scripts/package_pages.py
```

`package_pages.py` validates the public artifact and writes `_site/`. It publishes only the rendered book and seven reviewed supporting reports. Raw traces, prompts, machine-tool evidence, private run logs and source-generation scripts are not included in the Pages artifact. References to unpublished files remain labeled catalog entries without broken download links.

## Private repository and GitHub Pages

The repository must remain **private**. The supplied workflow deploys `_site/` using GitHub Actions; it never uploads the repository root as a site artifact.

**Repository privacy and site privacy are separate.** Ordinary GitHub Pages sites are public, including sites sourced from private repositories. Pages for a private personal repository requires an eligible paid GitHub plan; private site access requires an eligible organization configuration. The publishing script never makes the repository public as a workaround. See [GitHub's Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).

With a working authenticated GitHub CLI, run from PowerShell:

```powershell
.\scripts\publish.ps1 -RepositoryName inference-engine-textbook
```

The script checks authentication, prepares and validates the public artifact, initializes/commits this workspace if needed, creates a **new private repository** under the authenticated account with `gh`, pushes `main`, configures Pages to use Actions and starts the deployment. It stops on errors or an existing repository name rather than overwriting another project. If account policy or plan eligibility prevents Pages, the private repository is retained.

After a successful upload:

```powershell
gh run list --workflow pages.yml
gh run watch
gh api repos/OWNER/inference-engine-textbook/pages --jq .html_url
```

The committed HTML is the deployment input. Rebuild it locally before pushing manuscript changes; the Pages job does not install Pandoc or regenerate research content.

## Repository contents

- `research/textbook/` — book source, rendered HTML, teaching supplements and renderer.
- `research/*.md` — reviewed research, audit notes and retained editorial drafts.
- `research/prompts/` — research delegation prompts, stored in the private repository only.
- `scripts/` — restricted Pages packager and GitHub CLI publishing script.
- `.github/workflows/pages.yml` — automatic deployment on pushes to `main` and manual dispatch.

Large execution traces remain local and ignored by Git. The inventory preserves their historical filenames and roles; a cloned repository does not contain those archives.
