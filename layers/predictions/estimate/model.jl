using Revise
using PyCall
using Serialization

datetime = pyimport("datetime")
np = pyimport("numpy")
cd("src")
root = @__DIR__
push!(pyimport("sys")."path", root)
push!(pyimport("sys")."path", joinpath(root, "layers", "predictions"))
LoadXY = pyimport("load_xy").LoadXY

using Dates
using DataFrames
using DataStructures
using MLJ
import ScikitLearn.CrossValidation: KFold
import TsDb: Client

# .strftime("%Y-%m-%d_%H%M%S")
get_ex(;sym="") = "ex$(Dates.now(Dates.UTC))-$(sym)"

#     self.window_aggregator_window = [int(2**i) for i in range(13)]
#     self.window_aggregator_func = ["sum"]
#     self.window_aggregators = [WindowAggregator(window, func) for (window, func) in product(self.window_aggregator_window, self.window_aggregator_func)]
#     self.tags = {}
#     self.sym = sym
#     self.ex = ex(sym)
#     self.df = None
#     self.boosters = defaultdict(list)
#     self.pred_label_t = {}
#     self.pred_label_ho = {}


function purge_overlap(v_ix_t, v_ix_cv; i=250)
    """
    In each iteration remove i periods from around test reducing total train index elements
    i should be derived somewhat intelligently ...
    """
    if v_ix_cv[1] == 1  # test is on left side
        return v_ix_t[i:end]
    elseif v_ix_cv[1] > v_ix_t[end]  # test is on right side
        return v_ix_t[1:end-i]
    else  # test is surrounded by train
        i_left_end = v_ix_cv[1] - 1
        ix_train_left = findfirst(x->x==i_left_end, v_ix_t)
        i_right_start = v_ix_cv[end] + 1
        ix_train_right = findfirst(x->x==i_right_start, v_ix_t)
        return vcat(
            v_ix_t[1:ix_train_left],
            v_ix_t[ix_train_right:end]
            )
    end
end


function train_eval_pairs(cv_purged::CVPurged, rows)
    indices = []
    for (train, test) in KFold(size(rows)[1], n_folds=cv_purged.nfolds)
        train_less_overlap = purge_overlap(train, test)
        push!(indices, (train_less_overlap, test))
    end
    return indices
end


function load_inputs(exchange, sym, start, stop, features=missing)
    load_xy = LoadXY(exchange, sym, 
        datetime.datetime.fromisoformat(string(start)),  
        datetime.datetime.fromisoformat(string(stop)), label_ewm_span="64min")
    load_xy.assemble_frames()
    df = load_xy.df
    df = DataFrame(Dict(vcat(
        [(:ts, Nanosecond.(df.index.astype(np.int64).values) + DateTime(1970))],
        [(Symbol(c), df.get(c).values) for c in df.columns]
        )))

    println("Load XY done.")
    # apply sampling and purging separately perhaps
    v_label = load_xy.ps_label.values
    select!(df, :ts, Not(:ts))  # sorting
    return (df, v_label)
end


function split_ho(df, v_label; ho_share=0.3)
    n_ho = round(Int, size(df)[1] * ho_share)
    n_t = size(df)[1] - n_ho
    return  df[1:n_t, :], 
            df[end-n_ho:end, :], 
            v_label[1:n_t],
            v_label[end-n_ho:end]
end

function store_return_attribution(df, sym, ex)
    # need to have ts in names of df. push into upsert function. can use if type is jl DataFrame, otherwise first col
    select!(df, :ts, Not(:ts))
    meta = Dict(
        "measurement_name" => "weights",
        "asset" => sym,
        "information" => "return_attribution_sample_weights",
        "ex" => ex,
    )
    Client.upsert(meta, df)
end

function return_attribution_sample_weights(v_label; return_amplifier=100)
    """
    instance with high absolute return change should get a much higher weight. ignore the noise.
    overflow / underflow: normalize weights to have the smallest weight a weight of 1. Also easier intuitively later when plotting.
    """
    v = abs.(v_label .- 1) .* return_amplifier .+ 1
    return v ./ minimum(v)
end


function train(df, y, sym, ex::String)
    """t, ho  ;   t -> t_cv test_cv"""
    booster = @load LGBMRegressor

    v_ix_t, v_ix_ho = partition(1:size(df)[1], 0.7, shuffle=false)
    df_t, df_ho = df[v_ix_t, :], df[v_ix_ho, :]
    y_t, y_ho = y[v_ix_t], y[v_ix_ho]

    v_weight_t = return_attribution_sample_weights(y_t)
    # geometric_mean()
    # cluster_sample_weight(50).\

    store_return_attribution(DataFrame(Dict(
        :ts => df[v_ix_t, :ts],
        :weight => v_weight_t
    )), sym, ex)

    estimator_params = Dict([
        (:metric, ["l2"]),  # MSE
        # ("objective", "quantile"),
        # ("alpha", quantile),
        # ( "verbosity", 0),
        (:learning_rate, 0.05),
        # ("early_stopping_round", 100),
        (:boosting, "gbdt"),
        (:num_iterations, 1000),
        (:device_type, "gpu"),
    ])

    estimator = booster(;estimator_params...)

    iterated_model = IteratedModel(
        model=estimator,
        resampling=CV(nfolds=5, nboundary=250),
        iteration_parameter=:(num_iterations),
        measures=l2,
        controls=[Step(1),
            Patience(15),
            NumberLimit(50)
        ],
        retrain=true,
        )
    mach = machine(iterated_model, select(df_t, Not(:ts)), y_t, v_weight_t)    
    fit!(mach)
    
    # push!(boosters[quantile], lgb_booster)
    # push!(quantile_booster_scores[quantile], lgb_booster.best_score["valid_0"])
    # push!(preds_t, DataFrame(lgb_booster.predict(df[ix_cv]), index=df[ix_cv].index))
    # push!(preds_ho, DataFrame(lgb_booster.predict(df_ho), index=df_ho.index))


    preds_t = []
    preds_ho = []
    quantile_booster_scores = DefaultDict([])
    pred_label_t = DefaultDict([])
    pred_label_ho = DefaultDict([])

    # tree = machine(estimator, df_t, y_t, v_weight_t)
    # for (train, test) in KFold(size(df)[1], n_folds=5)
    #     ix_train_cv = ix_t[train]
    #     ix_test_cv = ix_t[test]
    #     ix_train_cv = purge_overlap(ix_train_cv, ix_test_cv)

    #     df_train, df_cv = df[ix_train_cv], df[ix_test_cv]
    #     y_train_cv, y_test_cv = y[ix_train_cv], y[ix_test_cv]
    #     v_weight_train_cv, v_weight_test_cv = v_weight_t[ix_train_cv], v_weight_t[ix_test_cv]

    #     estimator = booster(;estimator_params...)
    #     tree = machine(estimator, df_train, y_train_cv, v_weight_train_cv)
        
    #     fit!(tree, rows=train)
        
    #     fitted_params(tree) |> pprint
    #     ŷ = predict(tree, rows=test)

    #     push!(boosters[quantile], lgb_booster)
    #     push!(quantile_booster_scores[quantile], lgb_booster.best_score["valid_0"])
    #     push!(preds_t, DataFrame(lgb_booster.predict(df[ix_cv]), index=df[ix_cv].index))
    #     push!(preds_ho, DataFrame(lgb_booster.predict(df_ho), index=df_ho.index))
    # end

    # preds_t = vcat(preds_t).groupby(level=0).mean()
    # preds_ho = vcat(preds_ho).groupby(level=0).mean()
    # pred_label_t[quantile] = preds_t.merge(ps_label, how="inner", right_index=True, left_index=True)
    # pred_label_ho[quantile] = preds_ho.merge(ps_label_ho, how="inner", right_index=True, left_index=True)
end

    # def best_k_elbow(self, k_max: int):
    #     println("Find optimal # k cluster using Elbow method")
    #     sum_squared_distances = []
    #     k = list(range(2, k_max))
    #     for i, num_clusters in enumerate(k):
    #         kmeans = MiniBatchKMeans(n_clusters=num_clusters,
    #                              # random_state=0,
    #                              # batch_size=6,
    #                              max_iter=1000).fit(self.df.values)
    #         sum_squared_distances.append(kmeans.inertia_)
    #     res = pd.Series(dict(zip(k, sum_squared_distances))).plot()
    #     plt.xlabel("Values of K")
    #     plt.ylabel("Sum of squared distances/Inertia")
    #     plt.title("Elbow Method For Optimal k")
    #     plt.show()

    # def best_k_silhouette(self, k_max: int, k_min: int = 2):
    #     println("Find optimal # k cluster using Silhouette score")

    #     def silhouette_score(values, cluster_labels):
    #         scores = []
    #         map_label2vec_internal = {label: values[np.where(cluster_labels == label)[0], :] for label in np.unique(cluster_labels)}
    #         map_label2vec_external = {label: values[np.where(cluster_labels != label)[0], :] for label in np.unique(cluster_labels)}
    #         for i, label in enumerate(cluster_labels):
    #             internal_distance = np.linalg.norm(map_label2vec_internal[label] - values[i], axis=1).mean()
    #             external_distance = np.linalg.norm(map_label2vec_external[label] - values[i], axis=1).mean()
    #             scores.append((external_distance - internal_distance) / max(internal_distance, external_distance))
    #         return np.mean(scores)
    #     k = list(range(k_min, k_max))
    #     silhouette_avg = []
    #     for i, num_clusters in enumerate(k):
    #         if i % 10 == 0:
    #             print(i)
    #         # initialise kmeans
    #         kmeans = MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000).fit(self.df.values)
    #         # silhouette score
    #         silhouette_avg.append(silhouette_score(self.df.values, kmeans.labels_))
    #     plt.plot(k, silhouette_avg)
    #     plt.xlabel("Values of K")
    #     plt.ylabel("Silhouette score")
    #     plt.title("Silhouette analysis For Optimal k")
    #     plt.show()
    #     # Plot Dispersity
    #     dct = dict(Counter(kmeans.labels_))
    #     tup_lst = [(k, v) for k, v in dct.items()]
    #     tup_lst = sorted(tup_lst, key=lambda tup: tup[1], reverse=True)
    #     plt.bar(list(range(len(tup_lst))), [tup[1] for tup in tup_lst])
    #     plt.xlabel("Cluster Label")
    #     plt.ylabel("Count states")
    #     plt.title("Points per cluster")
    #     plt.show()

    #     return k[silhouette_avg.index(pd.Series(silhouette_avg).bfill().ffill().min())]


function main()
    exchange = "bitfinex"
    sym = "ethusd"
    ex = get_ex(sym=sym)
    start = Date(2022, 2, 7)
    stop = Date(2022, 2, 13)
    # end_ = Date(2022, 9, 3)
    println("From $(start) to $(stop)")
    
    features = ["_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-1500|aggWindow-256|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-256|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-1500|aggWindow-1024|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-adausd|unit_size-1000|aggWindow-512|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-tick|unit_size-30|aggWindow-1024|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-min", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-solusd|unit_size-30|aggWindow-1024|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-1024|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-512|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-1024|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-btcusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-150000|aggWindow-2048|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-2048|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-tick|unit_size-15|aggWindow-128|aggAggregator-sum", "_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-min", "_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-32|aggAggregator-max", "_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean", "_measurement-order_book|_field-count_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-max", "_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-min", "_measurement-trade_bars|_field-imbalance_size|asset-solusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-4000|aggWindow-2048|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-solusd|unit_size-30|aggWindow-128|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-adausd|unit_size-1000|aggWindow-1024|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-min", "_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean", "_measurement-trade_bars|_field-imbalance_size|asset-solusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-4000|aggWindow-256|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-min", "_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-7000|aggWindow-512|aggAggregator-sum", "_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-min", "_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-1024|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-btcusd|exchange-bitfinex|information-imbalance|unit-btcusd|unit_size-5|aggWindow-512|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-max", "_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-max", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-1024|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-256|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-tick|unit_size-30|aggWindow-512|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean", "_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-16|aggAggregator-min", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-128|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-usd|unit_size-1000|aggWindow-2048|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-tick|unit_size-30|aggWindow-2048|aggAggregator-sum", "_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-usd|unit_size-1000|aggWindow-1024|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-solusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-4000|aggWindow-512|aggAggregator-sum", "_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-256|aggAggregator-sum", "_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean", "_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-mean", "_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-adausd|unit_size-1000|aggWindow-2048|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-solusd|unit_size-30|aggWindow-2048|aggAggregator-sum", "_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-min", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean", "_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-3000|aggWindow-2048|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-xrpusd|unit_size-10000|aggWindow-2048|aggAggregator-sum", "_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-min", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-tick|unit_size-30|aggWindow-256|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-btcusd|exchange-bitfinex|information-imbalance|unit-btcusd|unit_size-5|aggWindow-2048|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-usd|unit_size-1000|aggWindow-512|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-btcusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-300000|aggWindow-128|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-64|aggAggregator-mean", "_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-xrpusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-128|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-32|aggAggregator-max", "_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-mean", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-512|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-ethusd|exchange-bitfinex|information-imbalance|unit-usd|unit_size-75000|aggWindow-512|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-max", "_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean", "_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-max", "_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-min", "_measurement-trade_bars|_field-imbalance_size|asset-btcusd|exchange-bitfinex|information-imbalance|unit-btcusd|unit_size-5|aggWindow-1024|aggAggregator-sum", "_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-1024|aggAggregator-max", "_measurement-trade_bars|_field-imbalance_size|asset-ethusd|exchange-bitfinex|information-imbalance|unit-tick|unit_size-30|aggWindow-512|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-min", "_measurement-trade_bars|_field-sequence_direction|asset-adausd|exchange-bitfinex|information-sequence|unit-usd|unit_size-1000|aggWindow-256|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-tick|unit_size-15|aggWindow-512|aggAggregator-sum", "_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-tick|unit_size-10|aggWindow-256|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-adausd|exchange-bitfinex|information-imbalance|unit-adausd|unit_size-1000|aggWindow-256|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-128|aggAggregator-min", "_measurement-order_book|_field-size_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-64|aggAggregator-min", "_measurement-order_book|_field-count_net|asset-adausd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-256|aggAggregator-mean", "_measurement-order_book|_field-size_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-mean", "_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-mean", "_measurement-order_book|_field-count_net|asset-solusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-2048|aggAggregator-mean", "_measurement-order_book|_field-size_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-64|aggAggregator-mean", "_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-mean", "_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-xrpusd|unit_size-10000|aggWindow-256|aggAggregator-sum", "_measurement-order_book|_field-count_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-64|aggAggregator-min", "_measurement-order_book|_field-size_net|asset-ethusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_size_imbalance_net|unit-size_ewm_sum|aggWindow-8|aggAggregator-min", "_measurement-order_book|_field-count_net|asset-btcusd|delta_size_ratio-0.5|exchange-bitfinex|information-bid_buy_count_imbalance_net|unit-size_ewm_sum|aggWindow-512|aggAggregator-max", "_measurement-trade_bars|_field-sequence_direction|asset-btcusd|exchange-bitfinex|information-sequence|unit-usd|unit_size-300000|aggWindow-1024|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-xrpusd|exchange-bitfinex|information-imbalance|unit-xrpusd|unit_size-10000|aggWindow-512|aggAggregator-sum", "_measurement-trade_bars|_field-sequence_direction|asset-solusd|exchange-bitfinex|information-sequence|unit-solusd|unit_size-30|aggWindow-512|aggAggregator-sum", "_measurement-trade_bars|_field-imbalance_size|asset-ethusd|exchange-bitfinex|information-imbalance|unit-tick|unit_size-30|aggWindow-2048|aggAggregator-sum"]
    features = missing
    if false
        df, y = load_inputs(exchange, sym, start, stop, features)
        serialize("df.jls", df)
        serialize("y.jls", y)
    else
        df = deserialize("df.jls")
        y = deserialize("y.jls")
    end
    # if features != missing
    #     df = df[:, [replace(replace(f, "order_book" => "order book"), "trade_bars" => "trade bars") for f in features]]
    # end
    # inst.best_k_elbow(100)
    # inst.best_k_silhouette(50, 50)
    # k ==50 is okay. rather have more than fewer. only useful if actually disperse. means some k have
    # much larger count than others... need dispersity measure for each k like median cnt?
    train(df, y, ex, exchange)
    # save()
    # inst.to_disk()
    println("Done. Ex: $(ex)")
end