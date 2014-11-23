using FactCheck

import OGD
ogd = OGD


facts("OGD tests") do
    # X and y values for testing
    num = 1000
    X = rand(-2:2, num, 3) * 1.;
    y = zeros(num)
    for t in 1:num
        y[t] = sum(X[t, :]) + 0.1 * randn() > 0
    end
    X = map(string, X);

    clfa = ogd.OGDLRa(["eins", "zwei", "drei"], 0.02, 1000)
    ogd.fit(clfa, X[:100, :], y[:100])
    clfd = ogd.OGDLRd(["eins", "zwei", "drei"], 0.02)
    ogd.fit(clfd, X[:100, :], y[:100])
    
    context("fitting improves score") do
        # each fit iteration should improve score on train set
        for __ in 1:10
            y_prob = ogd.predict_proba(clfa, X)
            ll_before_a = ogd.logloss(y, y_prob)
            ogd.fit(clfa, X, y)
            y_prob = ogd.predict_proba(clfa, X)
            ll_after_a = ogd.logloss(y, y_prob)
            @fact ll_before_a => greater_than(ll_after_a)
            
            y_prob = ogd.predict_proba(clfd, X)
            ll_before_d = ogd.logloss(y, y_prob)
            ogd.fit(clfd, X, y)
            y_prob = ogd.predict_proba(clfd, X)
            ll_after_d = ogd.logloss(y, y_prob)
            @fact ll_before_d => greater_than(ll_after_d)
        end
    end
    
    context("models make same predictions") do
        # note: not true if hash collision, but should be
        # unlikely with this few features
        y_prob_a = ogd.predict_proba(clfa, X)
        y_prob_d = ogd.predict_proba(clfd, X)
        @fact y_prob_a => y_prob_d
    end
    
    context("weights learned correctly") do
        # we know the relative weights
        for feat in ["eins", "zwei", "drei"]
            @fact (clfd.w[feat * "__-2.0"] => 
                less_than(clfd.w[feat * "__-1.0"]))
            @fact (clfd.w[feat * "__-1.0"] => 
                less_than(clfd.w[feat * "__0.0"]))
            @fact (clfd.w[feat * "__0.0"] => 
                less_than(clfd.w[feat * "__1.0"]))
            @fact (clfd.w[feat * "__1.0"] => 
                less_than(clfd.w[feat * "__2.0"]))
        end
    end
    
    context("effect of alpha (learning rate)") do
        # higher learning rate -> greater (absolute) weights
        mean_abs_weights = Float64[]
        for alpha in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            clf = ogd.OGDLRd(["eins", "zwei", "drei"], alpha)
            ogd.fit(clf, X[:10, :], y[:10, :])
            append!(mean_abs_weights, [mean(abs(collect(values(clf.w))))])
        end
        @fact mean_abs_weights[1] => 0
        @fact diff(mean_abs_weights) .> 0 => repmat([true], 6)
    end
end