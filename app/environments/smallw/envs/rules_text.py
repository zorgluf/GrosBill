"""Short reminders of the special rules of every race and power (GUI tooltips).

Static data only, read by `render_web.py`. The wording follows the rules digest
of `docs/smallw_plan.md` (section 2.1) and the rulings recorded there, so it
describes what the engine actually does.
"""

from __future__ import annotations

from .classes import PowerId, RaceId

#: One reminder per race, keyed by :class:`RaceId`.
RACE_RULES: dict[RaceId, str] = {
    RaceId.AMAZONS: (
        '+4 tokens usable only for conquests: they are taken back at the end '
        'of each redeployment and return to hand next turn.'),
    RaceId.DWARVES: (
        '+1 coin per Mine region occupied, even in decline.'),
    RaceId.ELVES: (
        'Lose no token when one of your regions is conquered: all of them '
        'come back in hand.'),
    RaceId.GHOULS: (
        'Keep all your tokens when going in decline. In decline, they still '
        'conquer at the start of each turn, before your active race.'),
    RaceId.GIANTS: (
        '-1 token (min 1) to conquer a region adjacent to a Mountain region '
        'you occupy.'),
    RaceId.HALFLINGS: (
        'May enter the map anywhere. A Hole-in-the-Ground in each of the first '
        '2 regions conquered makes them immune to conquests and powers. The '
        'holes are removed on decline or when the region is abandoned.'),
    RaceId.HUMANS: (
        '+1 coin per Farmland region occupied.'),
    RaceId.ORCS: (
        '+1 coin per non-empty region conquered this turn.'),
    RaceId.RATMEN: (
        'No special rule: strength in numbers.'),
    RaceId.SKELETONS: (
        'At redeployment, +1 token from the tray for every 2 non-empty '
        'regions conquered this turn.'),
    RaceId.SORCERERS: (
        'Once per turn per opponent, replace a lone active enemy token in a '
        'region adjacent to your Sorcerers by a Sorcerer from the tray. '
        'Encampments, Holes, Heroes and the Dragon protect it; Lairs, '
        'Fortresses and Mountains do not.'),
    RaceId.TRITONS: (
        '-1 token (min 1) to conquer a Coastal region (bordering a sea or '
        'the lake).'),
    RaceId.TROLLS: (
        "A Troll's Lair in every region you occupy: +1 defence. It stays in "
        'decline and is removed when the region is abandoned or conquered.'),
    RaceId.WIZARDS: (
        '+1 coin per Magic Source region occupied.'),
}

#: One reminder per special power, keyed by :class:`PowerId`.
POWER_RULES: dict[PowerId, str] = {
    PowerId.ALCHEMIST: (
        '+2 coins at the end of each turn while active.'),
    PowerId.BERSERK: (
        'Roll the reinforcement die before each conquest: the roll lowers its '
        'cost (min 1).'),
    PowerId.BIVOUACKING: (
        '5 Encampments to place at redeployment, in any split: +1 defence '
        'each, never lost in attacks, may be moved every turn. Removed on '
        'decline.'),
    PowerId.COMMANDO: (
        '-1 token (min 1) on every conquest.'),
    PowerId.DIPLOMAT: (
        'At the end of the turn, pick an opponent whose active race you did '
        'not attack this turn: it cannot attack your active race until your '
        'next turn.'),
    PowerId.DRAGON_MASTER: (
        'Once per turn, conquer a region with a single token whatever its '
        'defence. The Dragon makes it immune until it moves. Removed on '
        'decline.'),
    PowerId.FLYING: (
        'Conquer any region except seas and the lake: adjacency and the '
        'border rule do not apply.'),
    PowerId.FOREST: (
        '+1 coin per Forest region occupied.'),
    PowerId.FORTIFIED: (
        'Once per turn, put a Fortress in one of your regions (max 1 per '
        'region, 6 in total): +1 defence, even in decline, and +1 coin per '
        'Fortress at turn end while active.'),
    PowerId.HEROIC: (
        'At the end of the turn, place the 2 Heroes in 2 of your regions: '
        'they are immune to conquests and powers. Removed on decline.'),
    PowerId.HILL: (
        '+1 coin per Hill region occupied.'),
    PowerId.MERCHANT: (
        '+1 coin per region occupied.'),
    PowerId.MOUNTED: (
        '-1 token (min 1) to conquer a Hill or Farmland region.'),
    PowerId.PILLAGING: (
        '+1 coin per non-empty region conquered this turn.'),
    PowerId.SEAFARING: (
        'Seas and the lake can be conquered like empty regions (cost 2) and '
        'stay yours in decline.'),
    PowerId.SPIRIT: (
        'This race, once in decline, does not count toward the limit of one '
        'declined race: you may keep a second one.'),
    PowerId.STOUT: (
        'May go in decline at the end of a normal turn, after scoring.'),
    PowerId.SWAMP: (
        '+1 coin per Swamp region occupied.'),
    PowerId.UNDERWORLD: (
        '-1 token (min 1) to conquer a Cavern region; all Cavern regions are '
        'adjacent to each other.'),
    PowerId.WEALTHY: (
        '+7 coins at the end of your first turn with this power.'),
}
