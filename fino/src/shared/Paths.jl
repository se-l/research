module Paths

using Dates

const LOG_FN = "log_{}.txt"
const SRC_PATH = joinpath(@__DIR__, "..")
const PROJECT_PATH = joinpath(SRC_PATH, "..")
const COMMON = joinpath(SRC_PATH, "shared")

# Environment variable (assumes PATH_TRADE is set; use get(ENV, "PATH_TRADE", default) if optional)
const PATH_TRADE = ENV["PATH_TRADE"]
const ANALYTICS = joinpath(PATH_TRADE, "Analytics")
const PATH_MODELS = joinpath(PATH_TRADE, "models")
const PATH_DATA = joinpath(PATH_TRADE, "data")
const PATH_IB = joinpath(PATH_TRADE, "ib")

const PATH_DATA_ALTERNATIVE = joinpath(PATH_DATA, "alternative")
const PATH_DATA_INTEREST_RATE = joinpath(PATH_DATA_ALTERNATIVE, "interest-rate")
const PATH_SYMBOL_PROPERTIES = joinpath(PATH_DATA, "symbol-properties")
const PATH_MARKET_HOURS = joinpath(PATH_DATA, "market-hours")
const PATH_ACTIVITY_REPORTS_YTD = joinpath(PATH_IB, "activityReportsYTD")
const PATH_ANALYSIS_FRAMES = joinpath(ANALYTICS, "analysis_frames")
const PATH_CALIBRATION = joinpath(ANALYTICS, "calibration")
const PATH_API_CACHE = joinpath(ANALYTICS, "api_cache")

const PATH_EARNINGS = joinpath(PATH_SYMBOL_PROPERTIES, "EarningsAnnouncements.json")
const PATH_DIVIDEND_YIELDS = joinpath(PATH_SYMBOL_PROPERTIES, "DividendYields.json")
const PATH_MARKET_HOURS_DATABASE = joinpath(PATH_MARKET_HOURS, "market-hours-database.json")

export LOG_FN, SRC_PATH, PROJECT_PATH, COMMON, PATH_TRADE, ANALYTICS,
PATH_MODELS, PATH_DATA, PATH_IB, PATH_DATA_ALTERNATIVE,
PATH_DATA_INTEREST_RATE, PATH_SYMBOL_PROPERTIES, PATH_MARKET_HOURS,
PATH_ACTIVITY_REPORTS_YTD, PATH_ANALYSIS_FRAMES, PATH_CALIBRATION,
PATH_API_CACHE, PATH_EARNINGS, PATH_DIVIDEND_YIELDS,
PATH_MARKET_HOURS_DATABASE, mkdir

"""
Create directories if they don't exist. Returns the path.
"""
function mkdir(path::AbstractString)
    mkpath(path)
    return path
end

end # module Paths
