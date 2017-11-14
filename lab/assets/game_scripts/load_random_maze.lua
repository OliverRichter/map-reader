local pickups = require 'common.pickups'
local custom_observations = require 'decorators.custom_observations'
local timeout = require 'decorators.timeout'

-- Loads a random maze with thick walls.
-- Apples are placed near the goal and spawn points away from it.
-- Timeout is set to 3 minutes.
local api = {}

function api:createPickup(className)
  return pickups.defaults[className]
end

function api:start(maze_size, seed, params)
  api._maze_size = maze_size
  api._seed = seed
end

function api:nextMap()
  return 'map_' .. api._maze_size .. '_' .. api._seed
end

custom_observations.decorate(api)
timeout.decorate(api, 5 * 60)

return api
