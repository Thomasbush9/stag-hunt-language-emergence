# Issue tracker: Local Markdown

Issues and PRDs for this repo live as Markdown files in `.scratch/`, which is
intentionally ignored by Git while the experiment is local and exploratory.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The PRD is `.scratch/<feature-slug>/PRD.md`
- Implementation issues are `.scratch/<feature-slug>/issues/<NN>-<slug>.md`
- Triage state is a `Status:` line near the top of an issue file.
- Comments append under a `## Comments` heading.

When a skill publishes an issue, it creates a file under the relevant feature
directory. When a skill fetches a ticket, it reads the referenced local file.

