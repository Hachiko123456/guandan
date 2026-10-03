# A03 Scheme B Action Protocol

Command: `python scripts/run_acceptance.py --stage A03`.

Implement LEAD_PLAY, FOLLOW_PLAY, TRIBUTE, RETURN, FAMILY/RANK/LENGTH/SUIT/CARD/WILD_ASSIGNMENT tokens and COMMIT. Prefixes must not mutate public state; dead prefixes, duplicate physical cards, illegal wild-to-joker assignments and overflow must be rejected. An independent complete-action enumerator must match reachable committed actions on reduced fixtures. Token and committed-step counters and zero prefix reward must be tested.
