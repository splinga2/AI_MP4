import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from random import shuffle
from random import uniform

input_filename = None
if(len(sys.argv) == 2):
	input_filename = sys.argv[1]

game_width = 1.0
game_height = 1.0
paddle_height = 0.2
paddle_speed = 0.04
actions = [0, 1, -1]
initial_state = (0.5, 0.5, 0.03, 0.01, 0.5 - (paddle_height/2), False)
write = sys.stdout.write

# Updates the environment
def update_state(state, action, bounces):
	ball_x = state[0]
	ball_y = state[1]
	velocity_x = state[2]
	velocity_y = state[3]
	paddle_y = state[4]
	is_past = state[5]
	reward = 0

	if(is_past):
		reward = -1
		return (state, reward, bounces)

	# Move paddle
	paddle_y += paddle_speed * actions[action]

	if(paddle_y < 0.0):
		paddle_y = 0.0
	if(paddle_y > 1.0 - paddle_height):
		paddle_y = 1.0 - paddle_height

	# Move ball
	new_ball_x = velocity_x + ball_x
	new_ball_y = velocity_y + ball_y

	if(new_ball_y < 0.0):
		new_ball_y = -new_ball_y
		velocity_y = -velocity_y
	if(new_ball_y > 1.0):
		new_ball_y = 2 - new_ball_y
		velocity_y = -velocity_y
	if(new_ball_x < 0.0):
		new_ball_x = -new_ball_x
		velocity_x = -velocity_x

	# Check for paddle hit
	if(new_ball_x >= 1.0):
		slope = (new_ball_y - ball_y) / (new_ball_x - ball_x)
		inter_y = slope * (1.0 - ball_x) + ball_y
		if(inter_y >= paddle_y and inter_y <= paddle_y + paddle_height):
			bounces += 1
			new_ball_x = 2.0 - new_ball_x
			velocity_x = -velocity_x + uniform(-0.015, 0.015)
			if(abs(velocity_x) < 0.03):
				if(velocity_x < 0.0):
					velocity_x = -0.03
				else:
					velocity_x = 0.03
			velocity_y = velocity_y + uniform(-0.03, 0.03)
			reward = 1
		else:
			reward = -1
			is_past = True

	if(velocity_y > 1.0):
		velocity_y = 1.0
	if(velocity_y < -1.0):
		velocity_y = -1.0
	if(velocity_x > 1.0):
		velocity_x = 1.0
	if(velocity_x < -1.0):
		velocity_x = -1.0

	new_state = (new_ball_x, new_ball_y, velocity_x, velocity_y, paddle_y, is_past)
	return (new_state, reward, bounces)

def discretize(state):
	ball_x = state[0]
	ball_y = state[1]
	velocity_x = state[2]
	velocity_y = state[3]
	paddle_y = state[4]
	is_past = state[5]

	#if(is_past):
	#	return 0

	ball_x = floor(11 * ball_x)
	#if(ball_x == 12):
	#	ball_x == 11
	ball_y = floor(11 * ball_y)
	#if(ball_y == 12):
	#	ball_y == 11

	if(velocity_x > 0.0):
		velocity_x = 1
	else:
		velocity_x = 0

	if(velocity_y >= 0.015):
		velocity_y = 2
	elif(velocity_y <= -0.015):
		velocity_y = 0
	else:
		velocity_y = 1

	paddle_y = floor(12 * paddle_y/(1.0 - paddle_height))
	if(paddle_y == 12):
		paddle_y = 11

	return(ball_x, ball_y, velocity_x, velocity_y, paddle_y, is_past)

def getIndex(state):
	ball_x = state[0]
	ball_y = state[1]
	velocity_x = state[2]
	velocity_y = state[3]
	paddle_y = state[4]
	is_past = state[5]
	if(is_past):
		return 0
	index = paddle_y
	index += velocity_y * 12
	index += velocity_x * 3 * 12
	index += ball_y * 2 * 3 * 12
	index += ball_x * 12 * 2 * 3 * 12
	return floor(index + 1)

R_plus = 2
Ne = 4
rand_num = .1
def exploration(u, n):
	if(n < Ne):
		return float('inf')
	else:
		return u
	'''r = uniform(0.0, 1.0)
	if(r < rand_num):
		return float('inf')
	else:
		return u'''

def maxAction(current_index):
	max_q = -float('inf')
	current_action = 0
	for a in range(3):
		val = exploration(Q[a][current_index], N[a][current_index])
		if(val > max_q):
			max_q = val
			current_action = a
	return current_action

# Learning structures
Q = [[0.0] * 10369 for i in range(3)]
N = [[0] * 10369 for i in range(3)]
gamma = 0.95
t = 1
state = initial_state

total_bounces = 0
total_games = 0
if(input_filename is None):
	# Train
	for i in range(500000):
		bounces = 0
		total_games += 1
		t = 1
		state = initial_state
		while(True):
			# Select action
			current_i = getIndex(discretize(state))
			current_action = maxAction(current_i)
			current_Q = Q[current_action][current_i]

			# Update N
			alpha = 250.0 / (250.0 + N[current_action][current_i]) 
			N[current_action][current_i] += 1

			# Get successor state
			next_state, reward, bounces = update_state(state, current_action, bounces)

			next_i = getIndex(discretize(next_state))
			next_action = maxAction(next_i)

			# Update Q
			Q[current_action][current_i] += alpha * (reward + gamma * Q[next_action][next_i] - current_Q)

			if(state[5] and next_state[5]):
				break

			state = next_state
			t += 1
		total_bounces += bounces
		#print("Bounces for game%d: %d" % (total_games, bounces))
		#print("Average bounces so far: %.3f" % (total_bounces / total_games))
		#input()
else:
	# Read input file
	with open(input_filename, 'r') as f:
		for a in range(3):
			for q in range(10369):
				Q[a][q] = float(f.readline())
		for a in range(3):
			for n in range(10369):
				N[a][n] = float(f.readline())

	for i in range(5000):
		bounces = 0
		total_games += 1
		t = 1
		state = initial_state
		while(True):
			# Select action
			current_i = getIndex(discretize(state))
			current_action = maxAction(current_i)
			current_Q = Q[current_action][current_i]

			# Update N
			alpha = 300.0 / (300.0 + N[current_action][current_i]) 
			N[current_action][current_i] += 1

			# Get successor state
			next_state, reward, bounces = update_state(state, current_action, bounces)

			next_i = getIndex(discretize(next_state))
			next_action = maxAction(next_i)

			# Update Q
			Q[current_action][current_i] += alpha * (reward + gamma * Q[next_action][next_i] - current_Q)

			if(state[5] and next_state[5]):
				break

			state = next_state
			t += 1
		total_bounces += bounces

print("Average bounces out of %d games: %.3f" % (total_games, (total_bounces / total_games)))

# Write results
if(input_filename is None):
	result_file = open("Q_results.txt", "w")
	for a in range(3):
		for q in Q[a]:
			result_file.write(str(q) + '\n')
	for a in range(3):
		for n in N[a]:
			result_file.write(str(n) + '\n')

'''
for l in range(100):
	state, reward = update_state(state, 2)
	print('Ball_x: %.3f Ball_y: %.3f' % (state[0], state[1]))

	disc_state = discretize(state)
	ball_x = disc_state[0]
	ball_y = disc_state[1]
	paddle_y = disc_state[4]

	print('____________')
	for j in range(12):
		write('|')
		for i in range(12):
			if(i == 11 and (paddle_y == j or paddle_y+1 == j)):
				write('[')
			elif(j == ball_y and i == ball_x):
				write('o')
			else:
				write(' ')
		write('\n')
	print('____________')

	print('\n')
	input()
	print('\n')'''
