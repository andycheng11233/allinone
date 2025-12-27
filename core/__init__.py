"""Core modules for the match analysis system."""

from .alias import (
    load_alias_table_from_json,
    save_alias_table_if_needed,
    rebuild_alias_lookup,
    resolve_alias,
    resolve_league_alias,
    upsert_alias,
    normalize_team_name,
    normalize_league,
    append_unalias_pending,
    alias_data,
    ALIAS_TABLE,
    LEAGUE_ALIAS_TABLE,
)

from .ai import (
    call_deepseek_api,
    normalize_parsed_data,
    has_meaningful_data_for_ai,
    build_ai_prompt_with_availability,
    parse_ai_json_response,
    perform_ai_analysis_for_match,
)

from .utils import (
    cprint,
    strip_accents,
    find_best_float_in_text,
    parse_decimal_tokens_from_concatenated,
    extract_ratings_or_average_from_text,
    logger,
    Fore,
    Style,
    AI_CACHE_PATH,
    HKJC_ODDS_PROCESSED_PATH,
    TITAN_STATS_PROCESSED_PATH,
    TITAN_STATS_BASE,
    BAD_STRING,
    MIN_SECTIONS_FOR_FULL,
)

from .matcher import (
    LiveMatchMatcher,
    name_similarity,
    league_bonus,
    token_overlap_score,
)

__all__ = [
    # Alias management
    'load_alias_table_from_json',
    'save_alias_table_if_needed',
    'rebuild_alias_lookup',
    'resolve_alias',
    'resolve_league_alias',
    'upsert_alias',
    'normalize_team_name',
    'normalize_league',
    'append_unalias_pending',
    'alias_data',
    'ALIAS_TABLE',
    'LEAGUE_ALIAS_TABLE',
    # AI functions
    'call_deepseek_api',
    'normalize_parsed_data',
    'has_meaningful_data_for_ai',
    'build_ai_prompt_with_availability',
    'parse_ai_json_response',
    'perform_ai_analysis_for_match',
    # Matcher
    'LiveMatchMatcher',
    'name_similarity',
    'league_bonus',
    'token_overlap_score',
    # Utilities
    'cprint',
    'strip_accents',
    'find_best_float_in_text',
    'parse_decimal_tokens_from_concatenated',
    'extract_ratings_or_average_from_text',
    'logger',
    'Fore',
    'Style',
    'AI_CACHE_PATH',
    'HKJC_ODDS_PROCESSED_PATH',
    'TITAN_STATS_PROCESSED_PATH',
    'TITAN_STATS_BASE',
    'BAD_STRING',
    'MIN_SECTIONS_FOR_FULL',
]
