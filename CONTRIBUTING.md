# Working Together

## Branching

- `main` should always work. Don't commit directly to it.
- Create a branch for each piece of work:
  ```bash
  git checkout main
  git pull
  git checkout -b <your-name>/<short-description>   # e.g. amber/kmeans-baseline
  ```
- Push your branch and open a Pull Request. At least one teammate reviews it before merging.

## Commits

- Commit small, focused changes with clear messages, e.g. `Add DBSCAN medoid focus selection`.
- Pull `main` often to avoid painful merge conflicts.

## Notebooks

- Name notebooks with a number and your initials so they sort and don't collide, e.g. `01_eda_ar.ipynb`.
- **Clear outputs before committing.** Large outputs bloat the repo and cause merge conflicts.
- Once code in a notebook is stable, move it into `src/` so both experiment arms share it.

## Reproducibility

- Use the shared random seeds for every experiment.
- Never touch the held-out test set until final evaluation.
- If you add a dependency, add it to `requirements.txt`.
