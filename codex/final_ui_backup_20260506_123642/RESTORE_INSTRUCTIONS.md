# Restore Instructions

Archive path:
/workspace/creative_os_final_ui_backup_20260506_123642.tar.gz

Patch path:
/workspace/creative_os_final_ui_diff_20260506_123642.patch

## Restore From Archive

From /workspace or its parent, extract the archive:

```bash
tar -xzf /workspace/creative_os_final_ui_backup_20260506_123642.tar.gz -C /workspace
```

The archive uses paths relative to /workspace, including agent_core/creative_os, scripts, tests, cli_cockpit_snapshots, codex, config, and status.

## Restore From Patch

If you only want to reapply code changes in a git checkout:

```bash
cd /workspace
git apply /workspace/creative_os_final_ui_diff_20260506_123642.patch
```

Some untracked files are included in the patch export as new-file diffs. If git apply rejects context due to later changes, restore those files from the tar.gz archive instead.

## Verify After Restore

```bash
python3 -m py_compile /workspace/agent_core/creative_os/textual_cockpit.py /workspace/scripts/creative_os_cockpit.py
python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v
python3 -m unittest /workspace/tests/test_creative_os_status.py -v
```
