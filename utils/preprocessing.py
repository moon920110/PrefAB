import os
import pandas as pd
import numpy as np

def get_intensity(df, columns):
    _df = df.copy()[columns]
    _df = (_df - _df.min()) / (_df.max() - _df.min())
    _df = _df.sum(axis=1)
    _df = (_df - _df.min()) / (_df.max() - _df.min())
    _df = _df.fillna(0)
    return _df


def get_diversity(df, columns):
    _df = df.copy()[columns]
    for col in columns:
        _df.loc[_df[col] > 0, col] = 1
        _df.loc[_df[col] <= 0, col] = 0
    return _df.sum(axis=1)


def cleaning_logs(config, logger):
	logger.info(f'Cleaning logs for {config["experiment"]["player"]}_{config["experiment"]["game"]}_{config["experiment"]["session"]}')

	player = config['experiment']['player']
	session = config['experiment']['session']
	game = config['experiment']['game']
	game_name = config['game_name'][game]
	dataset_name = config['experiment']['dataset_name']

	raw_path = os.path.join(config['data']['path'], 'raw_data', f'{player}_{game_name}_{session}.csv')
	clean_path = os.path.join(config['data']['path'], 'clean_data', f'{player}_{game_name}_{session}_clean.csv')

	cleaned_df = pd.DataFrame()
	session_df = pd.read_csv(raw_path)

	cleaned_df['time_stamp'] = session_df['timeStamp'] - session_df['timeStamp'].iloc[0]
	cleaned_df['player_id'] = player
	cleaned_df['session_id'] = session
	cleaned_df['genre'] = dataset_name
	cleaned_df['game'] = game
	start_time = pd.Timedelta(0)
	time_index = [start_time + pd.Timedelta(seconds=0.25 * i) for i in range(len(session_df))]
	cleaned_df['time_index'] = time_index
	cleaned_df['time_index'] = cleaned_df['time_index'].astype(str)
	cleaned_df['epoch'] = session_df['epoch']
	cleaned_df['engine_tick'] = session_df['tick']
	cleaned_df['key_press_count'] = get_intensity(session_df, ['keyPressCount'])
	cleaned_df['idle_time'] = get_intensity(session_df, ['idleTime'])
	cleaned_df['player_score'] = get_intensity(session_df, ['playerScore'])
	cleaned_df['player_delta_distance'] = get_intensity(session_df, ['playerDeltaDistance'])
	cleaned_df['player_delta_rotation'] = get_intensity(session_df, ['playerDeltaRotation'])
	cleaned_df['player_kill_count'] = get_intensity(session_df, ['playerKillCount'])
	cleaned_df['player_speed_x'] = get_intensity(session_df, ['playerSpeedX'])
	cleaned_df['player_speed_y'] = get_intensity(session_df, ['playerSpeedY'])
	cleaned_df['player_speed_z'] = get_intensity(session_df, ['playerSpeedZ'])
	cleaned_df['player_health'] = get_intensity(session_df, ['playerHealth'])
	cleaned_df['player_damaged'] = get_intensity(session_df, ['playerDamaged'])
	cleaned_df['player_shooting'] = get_intensity(session_df, ['playerShooting'])
	cleaned_df['player_projectile_count'] = get_intensity(session_df, ['playerProjectileCount'])
	cleaned_df['player_projectile_distance'] = get_intensity(session_df, ['playerProjectileDistance'])
	cleaned_df['reticle_delta_distance'] = get_intensity(session_df, ['reticleDeltaDistance'])
	cleaned_df['player_aim_at_enemy'] = get_intensity(session_df, ['playerAimAtEnemy'])
	cleaned_df['player_aim_at_destructible'] = get_intensity(session_df, ['playerAimAtDestructible'])
	cleaned_df['player_health_pickup'] = get_intensity(session_df, ['playerHealthPickup'])
	cleaned_df['visible_bot_count'] = get_intensity(session_df, ['botsVisible'])
	cleaned_df['bot_speed_x'] = get_intensity(session_df, ['botSpeedX'])
	cleaned_df['bot_speed_y'] = get_intensity(session_df, ['botSpeedY'])
	cleaned_df['bot_speed_z'] = get_intensity(session_df, ['botSpeedZ'])
	cleaned_df['bot_delta_distance'] = get_intensity(session_df, ['botDeltaDistance'])
	cleaned_df['bot_delta_rotation'] = get_intensity(session_df, ['botDeltaRotation'])
	cleaned_df['bot_health'] = get_intensity(session_df, ['botHealth'])
	cleaned_df['bot_damaged'] = get_intensity(session_df, ['botDamaged'])
	cleaned_df['bot_shooting'] = get_intensity(session_df, ['botShooting'])
	cleaned_df['bot_projectile_count'] = get_intensity(session_df, ['botProjectileCount'])
	cleaned_df['bot_projectile_player_distance'] = get_intensity(session_df, ['botProjectilePlayerDistance'])
	cleaned_df['bot_aim_at_player'] = get_intensity(session_df, ['botAimAtPlayer'])
	cleaned_df['pick_ups_visible'] = get_intensity(session_df, ['pickUpsVisible'])
	cleaned_df['pick_up_player_disctance'] = get_intensity(session_df, ['pickUpPlayerDistance'])
	cleaned_df['destructible_count'] = get_intensity(session_df, ['destructibleCount'])
	cleaned_df['objects_destroyed'] = get_intensity(session_df, ['objectsDestroyed'])
	cleaned_df['player_death'] = get_intensity(session_df, ['playerDeath'])

	# Assign General Features
	session_df['[general]timePassed'] = (session_df['timeStamp']
	                                     - session_df['timeStamp'].iloc[0]) \
	                                    * 1000
	cleaned_df['time_passed'] = get_intensity(session_df, ['[general]timePassed'])
	cleaned_df['input_intensity'] = get_intensity(session_df, ["keyPressCount"])
	session_df['[general]inputDiversity'] = session_df['keyPresses'] \
		.apply(lambda x: 0 if pd.isnull(x) else len(set(x.split('|'))))
	cleaned_df['input_diversity'] = get_intensity(session_df, ['[general]inputDiversity'])
	cleaned_df['activity'] = 1 - cleaned_df["idle_time"]
	cleaned_df['score'] = get_intensity(session_df, ["playerScore"])
	cleaned_df['bot_count'] = get_intensity(session_df, ["botsVisible"])
	cleaned_df['bot_diversity'] = session_df['botsVisible'] \
		.apply(lambda x: 0 if x == 0 else 1) \
		if dataset_name.lower() != "platform" else \
		session_df['botTypes'].apply(lambda x: 0 if pd.isnull(x) else len(set(x.split('|'))))
	cleaned_df['bot_movement'] = get_intensity(session_df, ["botDeltaDistance"])
	session_df['[general]playerMovement'] = session_df['playerDeltaDistance'] \
		if dataset_name.lower() != "shooter" else \
		session_df['playerDeltaDistance'] + session_df['reticleDeltaDistance']
	cleaned_df['player_movement'] = get_intensity(session_df, ['[general]playerMovement'])
	# Assessing Object Intensity and Diversity
	if dataset_name.lower() == 'platform':
		session_df['[general]objectIntensity'] = session_df['pickUpsVisible']
		session_df['[general]objectDiversity'] = session_df['pickUpTypes'].apply(
			lambda x: 0 if pd.isnull(x) else len(set(x.split('|'))))
	if dataset_name.lower() == 'shooter':
		session_df['[general]objectIntensity'] = get_intensity(session_df, ['pickUpsVisible',
		                                                                    'destructibleCount'])
		session_df['[general]objectDiversity'] = get_diversity(session_df, ['pickUpsVisible',
		                                                                    'destructibleCount'])
	if dataset_name.lower() == 'racing':
		session_df['[general]objectIntensity'] = get_intensity(session_df, ['visibleJumpCount',
		                                                                    'visibleSpeedBoostCount',
		                                                                    'visibleObstacleCount',
		                                                                    'visibleLoopCount'])
		session_df['[general]objectDiversity'] = get_diversity(session_df, ['visibleJumpCount',
		                                                                    'visibleSpeedBoostCount',
		                                                                    'visibleObstacleCount',
		                                                                    'visibleLoopCount'])
	cleaned_df['object_intensity'] = session_df['[general]objectIntensity']
	cleaned_df['object_diversity'] = get_intensity(session_df, ['[general]objectDiversity'])

	# Assign Event Intensity and Diversity
	if dataset_name.lower() == 'platform':
		session_df['[general]eventIntensity'] = get_intensity(session_df, ['playerIsCollidingAbove',
		                                                                   'playerIsCollidingLeft',
		                                                                   'playerIsCollidingRight',
		                                                                   'playerIsFalling',
		                                                                   'playerIsJumping',
		                                                                   'playerDamaged',
		                                                                   'playerShooting',
		                                                                   'playerHealthPickup',
		                                                                   'playerPointPickup',
		                                                                   'playerPowerPickup',
		                                                                   'playerBoostPickup',
		                                                                   'playerSlowPickup',
		                                                                   'playerKillCount',
		                                                                   'botIsFalling',
		                                                                   'botIsJumping',
		                                                                   'botDamaged',
		                                                                   'botShooting',
		                                                                   'botCharging',
		                                                                   'playerDeath']).fillna(0)
		session_df['[general]eventDiversity'] = get_diversity(session_df, ['playerIsCollidingAbove',
		                                                                   'playerIsCollidingLeft',
		                                                                   'playerIsCollidingRight',
		                                                                   'playerIsFalling',
		                                                                   'playerIsJumping',
		                                                                   'playerDamaged',
		                                                                   'playerShooting',
		                                                                   'playerHealthPickup',
		                                                                   'playerPointPickup',
		                                                                   'playerPowerPickup',
		                                                                   'playerBoostPickup',
		                                                                   'playerSlowPickup',
		                                                                   'playerKillCount',
		                                                                   'botIsFalling',
		                                                                   'botIsJumping',
		                                                                   'botDamaged',
		                                                                   'botShooting',
		                                                                   'botCharging',
		                                                                   'playerDeath']).fillna(0)
	if dataset_name.lower() == 'shooter':
		session_df['[general]eventIntensity'] = get_intensity(session_df, ['playerKillCount',
		                                                                   'playerHealing',
		                                                                   'playerDamaged',
		                                                                   'playerShooting',
		                                                                   'playerReloading',
		                                                                   'playerCrouching',
		                                                                   'playerSprinting',
		                                                                   'playerHealthPickup',
		                                                                   'botDamaged',
		                                                                   'botShooting',
		                                                                   'botReloading',
		                                                                   'objectsDestroyed',
		                                                                   'playerDeath']).fillna(0)
		session_df['[general]eventDiversity'] = get_diversity(session_df, ['playerKillCount',
		                                                                   'playerHealing',
		                                                                   'playerDamaged',
		                                                                   'playerShooting',
		                                                                   'playerReloading',
		                                                                   'playerCrouching',
		                                                                   'playerSprinting',
		                                                                   'playerHealthPickup',
		                                                                   'botDamaged',
		                                                                   'botShooting',
		                                                                   'botReloading',
		                                                                   'objectsDestroyed',
		                                                                   'playerDeath']).fillna(0)
	if dataset_name.lower() == 'racing':
		session_df['[general]eventIntensity'] = get_intensity(session_df, ['playerSpeedBoost',
		                                                                   'playerIsLooping',
		                                                                   'playerIsCrashing',
		                                                                   'playerIsOffRoad',
		                                                                   'playerIsMidAir',
		                                                                   'playerRespawn',
		                                                                   'botSpeedBoost',
		                                                                   'botIsLooping',
		                                                                   'botIsOffRoad',
		                                                                   'botIsCrashing',
		                                                                   'botRespawn']).fillna(0)
		session_df['[general]eventDiversity'] = get_diversity(session_df, ['playerSpeedBoost',
		                                                                   'playerIsLooping',
		                                                                   'playerIsCrashing',
		                                                                   'playerIsOffRoad',
		                                                                   'playerIsMidAir',
		                                                                   'playerRespawn',
		                                                                   'botSpeedBoost',
		                                                                   'botIsLooping',
		                                                                   'botIsOffRoad',
		                                                                   'botIsCrashing',
		                                                                   'botRespawn']).fillna(0)

	cleaned_df['event_intensity'] = session_df['[general]eventIntensity']
	cleaned_df['event_diversity'] = get_intensity(session_df, ['[general]eventDiversity'])

	columns = [
		"genre", "player_id", "session_id", "game", "time_index", "epoch", "time_stamp", "engine_tick", "arousal",
		"time_passed", "input_intensity", "input_diversity", "activity", "score", "bot_count", "bot_diversity",
		"bot_movement", "player_movement", "object_intensity", "object_diversity", "event_intensity", "event_diversity",
		"key_presses", "player_aim_target", "bot_damaged_by", "key_press_count", "idle_time", "player_score",
		"player_kill_count", "player_speed_x", "player_speed_y", "player_speed_z", "player_delta_distance",
		"player_delta_rotation", "player_health", "player_healing", "player_damaged", "player_shooting",
		"player_reloading", "player_projectile_count", "player_projectile_distance", "reticle_delta_distance",
		"player_crouching", "player_sprinting", "player_aim_at_enemy", "player_aim_at_destructible",
		"player_health_pickup", "visible_bot_count", "bot_speed_x", "bot_speed_y", "bot_speed_z", "bot_delta_distance",
		"bot_delta_rotation", "bot_health", "bot_damaged", "bot_shooting", "bot_reloading", "bot_projectile_count",
		"bot_projectile_player_distance", "bot_aim_at_player", "pick_ups_visible", "pick_up_player_disctance",
		"destructible_count", "objects_destroyed", "player_death", "player_tries_shoot_on_reload", "player_standing",
		"player_speed", "player_speed_boost", "player_is_grounded", "player_is_mid_air", "player_is_looping",
		"player_is_crashing", "player_is_off_road", "player_gas_pedal", "player_steering", "player_lap",
		"player_distance_to_way_point", "player_respawn", "bot_standing", "bot_score", "bot_speed", "bot_speed_boost",
		"bot_is_grounded", "bot_is_looping", "bot_is_off_road", "bot_is_crashing", "bot_gas_pedal",
		"bot_steering", "bot_lap", "bot_distance_to_way_point", "bot_player_distance",
		"bot_respawn", "visible_jump_count", "visible_speed_boost_count", "visible_obstacle_count", "visible_loop_count",
		"player_damaged_by", "bot_types", "pick_up_types", "player_has_collisions", "player_is_colliding_above",
		"player_is_colliding_below", "player_is_colliding_left", "player_is_colliding_right",
		"player_is_falling", "player_is_jumping", "player_point_pickup", "player_power_pickup", "player_boost_pickup",
		"player_slow_pickup", "player_has_powerup", "bot_has_collisions", "bot_is_colliding_above",
		"bot_is_colliding_below", "bot_is_colliding_left", "bot_is_colliding_right", "bot_is_falling",
		"bot_is_jumping", "bot_charging",
		]
	cleaned_df = cleaned_df.reindex(columns, axis=1)
	cleaned_df = cleaned_df.dropna(axis=1, how='any')

	cleaned_df.to_csv(clean_path, index=False)
	return cleaned_df


def integrate_arousal():
	pass


if '__main__' == __name__:
	cleaning_logs('../data/p1_topdown_s1.csv', 'p1', 's1')