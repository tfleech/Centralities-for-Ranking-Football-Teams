import copy
import numpy as np
from numpy import linalg as LA
from elo import rate_1vs1
import operator
import networkx as nx

def get_data():
    file = open("cf1990gms.txt", "rU")
    line = file.readline().strip()
    games = []
    game_obj = {
        'away_team': None,
        'home_team': None,
        'away_team_score': None,
        'home_team_score': None,
        'neutral_site': None
    }

    def isint(value):
      try:
        int(value)
        return True
      except ValueError:
        return False

    while line:
        line_items = [x for x in line.split(' ') if x != ''][1:]
        cur_game_obj = {
            'away_team': None,
            'home_team': None,
            'away_team_score': None,
            'home_team_score': None,
            'neutral_site': None
        }
        cur_game_obj['away_team'] = line_items.pop(0)
        away_team_score = line_items.pop(0)
        if isint(away_team_score):
            cur_game_obj['away_team_score'] = int(away_team_score)
        else:
            cur_game_obj['away_team'] += ' ' + away_team_score
            away_team_score = line_items.pop(0)
            if isint(away_team_score):
                cur_game_obj['away_team_score'] = int(away_team_score)
            else:
                cur_game_obj['away_team'] += ' ' + away_team_score
                cur_game_obj['away_team_score'] = int(line_items.pop(0))

        cur_game_obj['home_team'] = line_items.pop(0)
        home_team_score = line_items.pop(0)
        if isint(home_team_score):
            cur_game_obj['home_team_score'] = int(home_team_score)
        else:
            cur_game_obj['home_team'] += ' ' + home_team_score
            home_team_score = line_items.pop(0)
            if isint(home_team_score):
                cur_game_obj['home_team_score'] = int(home_team_score)
            else:
                cur_game_obj['home_team'] += ' ' + home_team_score
                cur_game_obj['home_team_score'] = int(line_items.pop(0))

        if len(line_items) != 1:
            cur_game_obj['neutral_site'] = True
        else:
            cur_game_obj['neutral_site'] = False

        games.append(cur_game_obj)
        line = file.readline()

    return games

def construct_loss_graph(data):
    MG=nx.DiGraph()
    edges = []
    for game in data:
        if game['home_team_score'] > game['away_team_score']:
            winning_team = game['home_team']
            losing_team = game['away_team']
            edge_weight = 1
            score_differential = game['home_team_score'] - game['away_team_score']
            if score_differential >= 16:
                edge_weight += .1
            edges.append((winning_team, losing_team, edge_weight))
        elif game['away_team_score'] > game['home_team_score']:
            winning_team = game['away_team']
            losing_team = game['home_team']
            edge_weight = 1.1
            score_differential = game['away_team_score'] - game['home_team_score']
            if score_differential >= 16:
                edge_weight += .1
            edges.append((winning_team, losing_team, edge_weight))
        else:
            edges.append((game['home_team'], game['away_team'], .75))
            edges.append((game['away_team'], game['home_team'], .75))

    MG.add_weighted_edges_from(edges)
    return MG

def construct_win_graph(data):
    MG=nx.DiGraph()
    edges = []
    for game in data:
        if game['home_team_score'] > game['away_team_score']:
            winning_team = game['home_team']
            losing_team = game['away_team']
            edge_weight = 1.1
            score_differential = game['home_team_score'] - game['away_team_score']
            if score_differential >= 16:
                edge_weight += .1
            edges.append((losing_team, winning_team, edge_weight))
        elif game['away_team_score'] > game['home_team_score']:
            winning_team = game['away_team']
            losing_team = game['home_team']
            edge_weight = 1
            score_differential = game['home_team_score'] - game['away_team_score']
            if score_differential >= 16:
                edge_weight += .1
            edges.append((losing_team, winning_team, edge_weight))
        else:
            edges.append((game['home_team'], game['away_team'], .75))
            edges.append((game['away_team'], game['home_team'], .75))
    MG.add_weighted_edges_from(edges)
    return MG

# print nx.degree(construct_win_graph(get_data()))

def calculate_pagerank():
    page_rank_win = nx.pagerank_numpy(construct_win_graph(get_data()), alpha=.25)
    page_rank_loss = nx.pagerank_numpy(construct_loss_graph(get_data()), alpha=.25)
    page_rank_total = {}
    for team, pr in page_rank_win.iteritems():
        page_rank_total[team] = page_rank_win[team]
        if team in page_rank_loss:
            page_rank_total[team] = page_rank_total[team] - page_rank_loss[team]
    page_rank_sorted = sorted([x for x in page_rank_total.items()], key = lambda x: x[1], reverse=True)
    for x in page_rank_sorted[0:10]:
        print(x)

def test_for_multiple_games():
    game_teams = [[x['away_team'], x['home_team']] for x in get_data()]
    while len(game_teams) > 0:
        game_team = game_teams.pop()
        for gt in game_teams:
            if game_team[0] in gt and game_team[1] in gt:
                print('test')

def calculate_katz_win_loss():
    katz_win = nx.katz_centrality_numpy(construct_win_graph(get_data()), alpha=.25)
    katz_loss = nx.katz_centrality_numpy(construct_loss_graph(get_data()), alpha=.25)
    katz_total = {}
    for team in katz_win.keys():
        katz_total[team] = katz_win[team]
        if team in katz_loss:
            katz_total[team] = katz_total[team] - katz_loss[team]
    katz_sorted = sorted([x for x in katz_total.items()], key = lambda x: x[1], reverse=True)
    for x in katz_sorted[0:10]:
        print(x)

def calculate_katz_win_only():
    katz_win = nx.katz_centrality_numpy(construct_win_graph(get_data()), alpha=.25)
    katz_sorted = sorted([x for x in katz_win.items()], key = lambda x: x[1], reverse=True)
    for x in katz_sorted[0:10]:
        print(x)

def Calc_Elo(games):
    teams = set(g['home_team'] for g in games)
    teams.update(g['away_team'] for g in games)
    Elo_Scores = {t:100 for t in teams}
    for g in games:
        #print(Elo_Scores[g['home_team']])
        if (g['home_team_score'] > g['away_team_score']):
            rating_one = Elo_Scores[g['home_team']]
            rating_two = Elo_Scores[g['away_team']]
            New_Scores = rate_1vs1(rating_one, rating_two)
            Elo_Scores[g['home_team']] = New_Scores[0]
            Elo_Scores[g['away_team']] = New_Scores[1]
        elif (g['home_team_score'] < g['away_team_score']):
            New_Scores = rate_1vs1(Elo_Scores[g['away_team']], Elo_Scores[g['home_team']])
            Elo_Scores[g['away_team']] = New_Scores[0]
            Elo_Scores[g['home_team']] = New_Scores[1]
        else:
            New_Scores = rate_1vs1(Elo_Scores[g['home_team']], Elo_Scores[g['away_team']], drawn=True)
            Elo_Scores[g['home_team']] = New_Scores[0]
            Elo_Scores[g['away_team']] = New_Scores[1]

    return Elo_Scores

calculate_katz_win_only()
#Scores = Calc_Elo(get_data())
#sorted_scores = sorted(Scores.items(), key=operator.itemgetter(1), reverse=True)
#print(sorted_scores[:10])