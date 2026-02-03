This workflow runs the Upstox token refresh manager on a schedule and optionally updates repository secrets with newly issued tokens.

How it works:
- The job sets environment variables from repository secrets and runs `python -m src.token_manager --auto`.
- The workflow extracts tokens from the repository `.env` file (if written by the manager) and, if `REPO_PAT` secret is provided, will update `UPSTOX_ACCESS_TOKEN` and `UPSTOX_REFRESH_TOKEN` secrets using `peter-evans/create-or-update-secret`.

Security notes:
- Do not add your `.env` to the repository; it should remain in `.gitignore`.
- If you allow the workflow to update secrets automatically, use a short-lived or limited-scope token for `REPO_PAT` and rotate it periodically.
