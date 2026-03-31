"""GraphQL query definitions for the start.gg API."""

# Search for Melee tournaments in a date range
TOURNAMENTS_BY_GAME = """
query TournamentsByVideogame(
    $perPage: Int!,
    $page: Int!,
    $videogameId: ID!,
    $afterDate: Timestamp,
    $beforeDate: Timestamp
) {
    tournaments(query: {
        perPage: $perPage
        page: $page
        sortBy: "startAt asc"
        filter: {
            past: true
            videogameIds: [$videogameId]
            afterDate: $afterDate
            beforeDate: $beforeDate
        }
    }) {
        pageInfo {
            total
            totalPages
            page
            perPage
        }
        nodes {
            id
            name
            slug
            numAttendees
            startAt
            endAt
            city
            addrState
            countryCode
            isOnline
            events {
                id
                name
                slug
                numEntrants
                videogame {
                    id
                }
            }
        }
    }
}
"""

# Get all sets from an event with player info, scores, seeds, and game-level data
EVENT_SETS = """
query EventSets($eventId: ID!, $page: Int!, $perPage: Int!) {
    event(id: $eventId) {
        id
        name
        numEntrants
        tournament {
            id
            name
        }
        sets(page: $page, perPage: $perPage, sortType: STANDARD) {
            pageInfo {
                total
                totalPages
                page
            }
            nodes {
                id
                displayScore
                winnerId
                round
                fullRoundText
                completedAt
                totalGames
                state
                games {
                    id
                    orderNum
                    winnerId
                    stage {
                        id
                        name
                    }
                    selections {
                        entrant {
                            id
                        }
                        character {
                            id
                            name
                        }
                    }
                }
                slots {
                    seed {
                        seedNum
                    }
                    standing {
                        placement
                        stats {
                            score {
                                value
                            }
                        }
                    }
                    entrant {
                        id
                        name
                        initialSeedNum
                        participants {
                            player {
                                id
                                gamerTag
                            }
                        }
                    }
                }
                phaseGroup {
                    phase {
                        name
                    }
                }
            }
        }
    }
}
"""
