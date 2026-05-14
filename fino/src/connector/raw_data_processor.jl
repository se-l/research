using PyCall
using Dates
using Distributed

include("RawDataConfig.jl")

# ── Python stdlib / third-party ───────────────────────────────────────────────
const os = pyimport("os")
const shutil = pyimport("shutil")
const subprocess_py = pyimport("subprocess")
const zipfile = pyimport("zipfile")
const pd = pyimport("pandas")
const tempfile = pyimport("tempfile")
const itertools = pyimport("itertools")

# ── Your Python packages ──────────────────────────────────────────────────────
const connector_constants = pyimport("connector.constants")
const ib_enums = pyimport("connector.ib.enums")
const gen_oi = pyimport("connector.ib.generate_open_interest_files")
const tick2qc = pyimport("connector.ib.tick2qc_bar")
const upsample_equity_mod = pyimport("connector.ib.upsample_qc_equity_bars")
const upsample_option_mod = pyimport("connector.ib.upsample_qc_option_bars")
const options_helper = pyimport("options.helper")
const options_enums = pyimport("options.types.enums")
const shared_logger = pyimport("shared.modules.logger")
const shared_paths = pyimport("shared.paths")

const file_root = connector_constants.file_root
const file_root_live = connector_constants.file_root_live
const dt_fmt_ymd = connector_constants.dt_fmt_ymd
const TradeType = ib_enums.TradeType
const Resolution = ib_enums.Resolution
const SecurityType = options_enums.SecurityType
const TickType = options_enums.TickType
const add_trade_days = options_helper.add_trade_days
const Paths = shared_paths.Paths
const upsample_ticks_with_args = tick2qc.upsample_ticks_with_args
const upsample_equity_bars = upsample_equity_mod.upsample_equity_bars
const upsample_option_bars = upsample_option_mod.upsample_option_bars
const gen_openinterest_files = gen_oi.gen_openinterest_files

# ── ps_exec ───────────────────────────────────────────────────────────────────
function ps_exec(commands::Vector{String}; daemon::Bool = false)
    p = subprocess_py.Popen(
        ["powershell", "-ExecutionPolicy", "ByPass", "-Command", join(commands, " ")],
        shell = false,
        stdout = subprocess_py.PIPE,
        stderr = subprocess_py.PIPE,
    )
    if !daemon
        while true
            line = p.stdout.readline()
            isempty(line) && break
            println(line.decode("utf-8").strip())
        end
    end
end

# ── aws_login ─────────────────────────────────────────────────────────────────
    aws_login() = ps_exec(["aws sso login;"])

# ── aws_upload ────────────────────────────────────────────────────────────────
function aws_upload()
    ps_exec([
        "aws s3 sync C:/repos/trade/data/equity s3://sebtradedata/derivatives/equity;",
        "aws s3 sync C:/repos/trade/data/option  s3://sebtradedata/derivatives/option;",
    ])
end

# ── copy_live2prod ────────────────────────────────────────────────────────────
"""Copy all .zip files starting with `yyyyMMdd` from dataLive to data."""
function copy_live2prod(yyyymmdd::String)
    for (root, _, files) in os.walk(file_root_live)
        root_data = replace(root, "dataLive" => "data")
        for fn in files
            if startswith(fn, yyyymmdd) && endswith(fn, ".zip")
                dest = os.path.join(root_data, fn)
                println("Copying file $(os.path.join(root, fn)) to $dest")
                shutil.copy(os.path.join(root, fn), dest)
            end
        end
    end
end

# ── clean_up ──────────────────────────────────────────────────────────────────
"""Delete all files from root folder that start with `yyyyMMdd`."""
function clean_up(yyyymmdd::String)
    for (root, _, files) in os.walk(file_root)
        for fn in files
            if startswith(fn, yyyymmdd)
                path = os.path.join(root, fn)
                println("Removing file $path")
                os.remove(path)
            end
        end
    end
end

# ── upsample_with_args ────────────────────────────────────────────────────────
function upsample_with_args(args::Tuple)
    sec_type, market, resolution_from, resolution_to, symbol, trade_type, start_date, end_date = args
    if sec_type == SecurityType.option
        upsample_option_bars(sec_type, market, resolution_from, resolution_to, symbol, trade_type, start_date, end_date)
    elseif sec_type == SecurityType.equity
        upsample_equity_bars(sec_type, market, resolution_from, resolution_to, symbol, start_date, end_date)
    end
end

function upsample_with_args_lst(args_lst::Vector)
    for args in args_lst
        upsample_with_args(args)
    end
end

# ── find_empty_option_data_days ───────────────────────────────────────────────
"""
Occasionally entire days contain option .csv files without any data (likely a
download error). Find, delete and flag for redownload.
"""
function find_empty_option_data_days(configs::Vector{RawDataConfig})
    to_del = String[]
    for (dir_, folders, fns) in os.walk(raw"D:\\trade\\data\\option\\usa\\tick")
        !isempty(folders) && continue
        ticker = uppercase(split(dir_, "\\")[end])
        println("Checking $ticker")
        for fn in filter(f -> endswith(f, ".zip"), fns)
            path = os.path.join(dir_, fn)
            dt = Dates.Date(fn[1:8], dateformat"yyyymmdd")
            println("Checking $dt at $path")
            try
                archive = pyimport("zipfile").ZipFile(path, "r")
                if all(f.file_size == 0 for f in archive.filelist)
                    println(path)
                    push!(to_del, path)
                end
                archive.close()
            catch e
                println("Error: $e $path")
                push!(to_del, path)
            end
        end
    end
    for path in to_del
        println("Removing $path")
        os.remove(path)
    end
end

# ── rm_tmp_files ──────────────────────────────────────────────────────────────
function rm_tmp_files()
    for (dir_, _, fns) in os.walk(raw"D:\trade\data")
        for fn in filter(f -> endswith(f, ".tmp"), fns)
            path = os.path.join(dir_, fn)
            println("Removing $path")
            os.remove(path)
        end
    end
end

# ── rm_iv_quote_trade_tmp_files ───────────────────────────────────────────────
function rm_iv_quote_trade_tmp_files()
    for (dir_, _, fns) in os.walk(raw"D:\trade\data")
        for fn in fns
            if ("iv_quote" in fn || "iv_trade" in fn) && !("nke" in fn)
                path = os.path.join(dir_, fn)
                println("Removing $path")
                # os.remove(path)
            end
        end
    end
end

# ── rm_duplicate_zip_entry_names ──────────────────────────────────────────────
"""
Recursively walk `directory`, and for each .zip file remove duplicate entries,
keeping the largest copy of each name.
"""
function rm_duplicate_zip_entry_names(directory::String)
    for (root, _, files) in os.walk(directory)
        for file in filter(f -> endswith(f, ".zip"), files)
            zip_path = os.path.join(root, file)
            try
                archive = zipfile.ZipFile(zip_path, "r")

                name_to_entries = Dict{String, Vector}()
                for entry in archive.filelist
                    push!(get!(name_to_entries, entry.filename, []), entry)
                end

                duplicates = filter(p -> length(p.second) > 1, name_to_entries)
                if !isempty(duplicates)
                    println("Processing $zip_path with duplicates: $(collect(keys(duplicates)))")
                    to_remove = String[]
                    for (name, entries) in duplicates
                        sizes = sort([(e, e.file_size) for e in entries]; by = x -> x[2], rev = true)
                        println("  Duplicate '$name' sizes: $([(e.filename, s) for (e, s) in sizes])")
                        for i in 2:length(sizes)
                            push!(to_remove, sizes[i][1].filename)
                        end
                    end

                    if !isempty(to_remove)
                        tmp = tempfile.NamedTemporaryFile(delete = false, suffix = ".zip")
                        tmp_path = tmp.name
                        tmp.close()

                        new_archive = zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED)
                        for entry in archive.filelist
                            if entry.filename ∉ to_remove
                                data = archive.open(entry.filename).read()
                                new_archive.writestr(entry.filename, data)
                            end
                        end
                        new_archive.close()
                        shutil.move(tmp_path, zip_path)
                        println("  Removed smaller duplicates from $zip_path: $to_remove")
                    end
                end
                archive.close()
            catch e
                println("Error processing $zip_path: $e")
            end
        end
    end
end

# ── deleting_tmp_file ─────────────────────────────────────────────────────────
function deleting_tmp_file(directory::String)
    for (root, _, files) in os.walk(directory)
        for file in filter(f -> endswith(f, ".tmp"), files)
            path = os.path.join(root, file)
            println("Deleting tmp file: $path")
            os.remove(path)
        end
    end
end

# ── is_bad_zip ────────────────────────────────────────────────────────────────
function is_bad_zip(zip_path::String)::Bool
    try
        zf = zipfile.ZipFile(zip_path, "r")
        first_bad = zf.testzip()
        if !isnothing(first_bad) && first_bad != nothing
            zf.close()
            return true
        end
        for info in zf.infolist()
            f = zf.open(info, "r")
            while !isempty(f.read(1024 * 1024))
            end
        end
        zf.close()
    catch
        return true
    end
    return false
end
#
## ── scan_root_for_bad_zip_members ─────────────────────────────────────────────
#function scan_root_for_bad_zip_members(directory::String; start_from = nothing, remove::Bool = false)
#    zip_files = String[
#joinpath(root, file)
#for (root, _, files) in os.walk(directory)
#for file in files
#if endswith(file, ".zip")
#]
#
#if !isnothing(start_from)
#idx = findfirst(==(start_from), zip_files)
#isnothing(idx) && return
#zip_files = zip_files[idx:end]
#end
#
#total = length(zip_files)
#if total == 0
#println("No ZIP files found under $directory")
#return
#end
#
#scanned = Threads.Atomic{Int}(0)
#Threads.@threads for file_path in zip_files
#bad = is_bad_zip(file_path)
#n = Threads.atomic_add!(scanned, 1) + 1
#pct = (n / total) * 100
#try
#print("\rScanning $n/$total ($(round(pct; digits = 1))%) : $file_path ")
#catch e
#shared_logger.info(string(e))
#end
#if bad
#println("\nDeleting zip file with bad member: $file_path\n")
#remove && os.remove(file_path)
#end
#end
#println()
#end

# ── process_ticks_to_bars ─────────────────────────────────────────────────────
function process_ticks_to_bars(
    configs::Vector{RawDataConfig};
    skip_zip::Bool = true,
    n_processes::Int = 16,
    market::String = "usa",
    security_types = (SecurityType.equity, SecurityType.option)
)
    arg_list = []
    for cfg in configs
        for dt in pd.date_range(cfg.start, cfg.stop)
            for security_type in security_types
                for tick_type in [TickType.quote, TickType.trade]
                    for resolution in [Resolution.second, Resolution.minute]
                        for symbol in split(cfg.tickers, ',')
                            push!(arg_list, (
                                security_type, market, resolution, strip(symbol) |> lowercase,
                                tick_type, Dates.format(dt, "yyyymmdd"),
                                Dates.format(dt, "yyyymmdd"), true, skip_zip,
                            ))
                        end
                    end
                end
            end
        end
    end

    processed = Threads.Atomic{Int}(0)
    total = length(arg_list)

    addprocs(min(n_processes, total))
    pmap(arg_list) do args
        upsample_ticks_with_args(args...)
        n = Threads.atomic_add!(processed, 1) + 1
        pct = (n / total) * 100
        print("\rprocess_ticks_to_bars(): $(round(pct; digits = 1))%")
    end
end

# ── process_hour_daily_upsample ───────────────────────────────────────────────
function process_hour_daily_upsample(
configs::Vector{RawDataConfig};
n_processes::Int = 16,
security_types = (SecurityType.equity, SecurityType.option),
)
arg_list = []
for cfg in configs
for symbol in split(cfg.tickers, ',')
sym = strip(symbol) |> lowercase
for res in [Resolution.hour, Resolution.daily]
for security_type in security_types
for trade_type in [TradeType.trade, TradeType.quote]
push!(arg_list, (
security_type, "usa", Resolution.minute, res, sym,
trade_type,
Dates.format(cfg.start, "yyyymmdd"),
Dates.format(cfg.stop, "yyyymmdd"),
))
end
end
end
end
end

# Group by symbol (index 5, i.e. args[5])
sort!(arg_list; by = a -> a[5])
grouped = [collect(grp) for (_, grp) in itertools.groupby(arg_list, a -> a[5])]

processed = Threads.Atomic{Int}(0)
total = length(grouped)

addprocs(n_processes)
pmap(grouped) do args_group
upsample_with_args_lst(args_group)
n = Threads.atomic_add!(processed, 1) + 1
pct = (n / total) * 100
print("\rprocess_hour_daily_upsample(): $(round(pct; digits = 1))%")
end
end

# ── process_open_interest ─────────────────────────────────────────────────────
function process_open_interest(configs::Vector{RawDataConfig})
total = length(configs)
processed = 0
for cfg in configs
for resolution in [Resolution.tick, Resolution.minute, Resolution.second, Resolution.daily]
for sym in split(cfg.tickers, ',')
gen_openinterest_files(
sec_type = SecurityType.option,
market = "usa",
resolution = resolution,
symbol = strip(sym),
start_date = Dates.format(cfg.start, "yyyymmdd"),
)
end
end
processed += 1
print("\r$(round((processed / total) * 100; digits = 1))%")
end
end

# ── transform ─────────────────────────────────────────────────────────────────
function transform(configs::Vector{RawDataConfig}; skip_zip::Bool = false)
process_ticks_to_bars(configs; skip_zip)
process_hour_daily_upsample(configs)
process_open_interest(configs)
end

# ── move_live_data ────────────────────────────────────────────────────────────
function move_live_data(dt_in::Date; rev::Bool = false)
dt = Dates.format(dt_in, "yyyymmdd")
src_root = rev ? replace(file_root, "data" => "dataLive") : file_root
for (directory, _, files) in os.walk(src_root)
for fn in filter(f -> startswith(f, dt), files)
println("Moving $(os.path.join(directory, fn)) to live data folder")
target_folder = rev ?
replace(directory, "dataLive" => "data") :
replace(directory, "data" => "dataLive")
os.makedirs(target_folder; exist_ok = true)
src = os.path.join(directory, fn)
dst = os.path.join(target_folder, fn)
os.path.exists(dst) ? os.remove(src) : os.rename(src, dst)
end
end
end

# ── upsample_iv ───────────────────────────────────────────────────────────────
function upsample_iv(configs::Vector{RawDataConfig})
for cfg in configs
for symbol in split(cfg.tickers, ',')
sym = strip(symbol) |> lowercase
for trade_type in [TradeType.iv_quote, TradeType.iv_trade]
upsample_with_args((
SecurityType.option, "usa", Resolution.second, Resolution.daily, sym,
trade_type,
Dates.format(cfg.start, "yyyymmdd"),
Dates.format(cfg.stop, "yyyymmdd"),
))
end
end
end
end

# ── cp2local ──────────────────────────────────────────────────────────────────
function cp2local(sym::String)
src_dir = file_root
path_data = Paths.path_data
fn2copy = (fn, dir_name) -> fn in ("market-hours-database.json", "interest-rate.csv") ||
sym in fn || dir_name in (sym, "symbol-properties")

for (directory, _, files) in os.walk(src_dir)
dir_name = basename(directory)
for fn in files
fn2copy(fn, dir_name) || continue
tgt = os.path.join(directory, fn)
dst = os.path.join(replace(directory, src_dir => path_data), fn)
tgt_dir = dirname(dst)
os.makedirs(tgt_dir; exist_ok = true)
if ("minute" in tgt || Resolution.second in tgt || "tick" in tgt) && os.path.exists(dst)
continue
end
shutil.copy(tgt, dst)
println("Copy $tgt -> $dst")
end
end
end

# ── ps_export_implied_volatility_toolbox ──────────────────────────────────────
"""Non-docker version of the toolbox command. Not in use."""
function ps_export_implied_volatility_toolbox(start::Date, stop::Date, tickers::String)
start_str = Dates.format(start, "yyyymmdd")
stop_str = Dates.format(stop, "yyyymmdd")
ps_exec([
raw"cd C:\repos\quantconnect\Lean\Toolbox\bin\Release;",
"echo \$pwd;",
".\\QuantConnect.Toolbox.exe --app=ive --tickers=$tickers --from-date=$(start_str)-00:00:00 --to-date=$(stop_str)-00:00:00 --n-clients=32;",
])
end

# ── stubs ─────────────────────────────────────────────────────────────────────
scan_for_missing_tick_csv_entries() = nothing
scan_for_missing_bar_csv_entries(sym, sec_type, start, stop)::Nothing = nothing