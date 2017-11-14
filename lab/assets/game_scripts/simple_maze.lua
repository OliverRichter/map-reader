local maze_gen = require 'dmlab.system.maze_generation'
local tensor = require 'dmlab.system.tensor'
local random = require 'common.random'
local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local custom_observations = require 'decorators.custom_observations'
local timeout = require 'decorators.timeout'
local screen_message = require 'common.screen_message'

-- Generates a random maze with thick walls.
-- Apples are placed near the goal and spawn points away from it.
-- Timeout is set to 3 minutes.
local api = {}


function api:commandLine(oldCommandLine)
  return make_map.commandLine(oldCommandLine)
end

function api:createPickup(className)
  return pickups.defaults[className]
end

function api:start(episode, seed, params)
  local rows, cols = 7,7
  local maze = maze_gen.MazeGeneration{height = rows, width = cols }
  local entityLayer = '*******\n***AF**\n***A*F*\n*AAPAA*\n*F*A***\n**GA***\n*******'
  print('Maze Generated:')
  print(entityLayer)


  local map = entityLayer
  api._maze_name = make_map.makeMap('map_bottom_edge',entityLayer, maze:variationsLayer())

  return map
end

function api:nextMap()

  return api._maze_name
end

custom_observations.decorate(api)
timeout.decorate(api, 3 * 60)

return api
