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

function api:start(episode, seed, params)
  local map_names = {'map_top','map_left','map_bottom','map_right' }
  --local map_names = {'map_top_edge','map_left_edge','map_bottom_edge','map_right_edge' }
  api._map = map_names[seed+1]
end

function api:nextMap()
  return api._map
end

custom_observations.decorate(api)
timeout.decorate(api, 1 * 10)

return api
