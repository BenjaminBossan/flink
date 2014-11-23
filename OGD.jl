module OGD


type OGDLRd
    cols
    alpha
    w
end

OGDLRd(cols, alpha) = OGDLRd(cols, alpha, sizehint(Dict{ASCIIString, Float64}(), 2^20))


type OGDLRa
    cols
    alpha
    ndims
    w
end

OGDLRa(cols, alpha, ndims) = OGDLRa(cols, alpha, ndims, zeros(Float64, ndims))


function logloss(y_true, y_pred)
    epsilon = 1e-15
    y_pred = max(epsilon, y_pred)
    y_pred = min(1 - epsilon, y_pred)
    ll = sum(y_true .* log(y_pred) +
        (1 - y_true) .* log (1 - y_pred))
    ll = - ll / length(y_true)
end


function sigmoid(x):
    1 ./ (1 + exp(-x))
end


function get_x(clf::OGDLRd, x::Array{ASCIIString, 1})
    m = size(x, 1)
    xt = Array(ASCIIString, m + 1)
    xt[1] = "BIAS"
    for i in 1:m
        xt[i + 1] = string(clf.cols[i]) * "__" * string(x[i])
    end
    return xt
end

function get_x(clf::OGDLRa, x::Array{ASCIIString, 1})
    m = size(x, 1)
    xt = Array(Int64, m + 1)
    xt[1] = int(hash("BIAS") % clf.ndims)
    for i in 1:m
        elem = string(clf.cols[i]) * "__" * string(x[i])
        xt[i + 1] = int(hash(elem) % clf.ndims)
    end
    return xt
end


function get_w(clf::OGDLRd, x::Array{ASCIIString, 1})
    [get(clf.w, xi, 0) for xi in x]
end


function get_w(clf::OGDLRd, x::ASCIIString)
    get(clf.w, x, 0.)
end


function get_w(clf::OGDLRa, x::Array{Int64, 1})
    [clf.w[xi] for xi in x]
end


function get_w(clf::OGDLRa, x::Int64)
    clf.w[x]
end


function get_p(clf, xt)
    wTx = sum(get_w(clf, xt))
    # wTx = max(min(wTx, 20.), -20.)
    sigmoid(wTx)
end


function get_grad(clf, pt, yt)
    pt - yt
end


function update(clf, xt, grad)
    delta_w = grad * clf.alpha
    for xi in xt
        clf.w[xi] = get_w(clf, xi) - delta_w
    end
    return clf
end


function fit(clf, X, y)
    n = size(X, 1)
    for t in 1:n
        yt = y[t]
        xt = get_x(clf, vec(X[t, :]))
        pt = get_p(clf, xt)
        grad = get_grad(clf, pt, yt)
        clf = update(clf, xt, grad)
    end
end


function predict_proba(clf, x)
    n = size(x)[1]
    y_prob = zeros(n)
    for t in 1:n
        xt = get_x(clf, vec(x[t, :]))
        y_prob[t] = get_p(clf, xt)
    end
    return y_prob
end


function predict(clf, x)
    y_prob = predict_proba(clf, x)
    y_pred = y_prob .> 0.5
end


function accuracy(y_true, y_pred)
    mean(y_true .== y_pred)
end

end