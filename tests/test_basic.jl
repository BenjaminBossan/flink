using FactCheck
import OGD
ogd = OGD

facts("OGD tests") do
    context("utils") do
        @fact ogd.sigmoid(0) => 0.5
        @fact ogd.sigmoid(-20) => roughly(0; atol=1e-6)
        @fact ogd.sigmoid(20) => roughly(1; atol=1e-6)
    end
    
    clfa = ogd.OGDLRa(["eins", "zwei", "drei"], 0.02, 1000)
    clfd = ogd.OGDLRd(["eins", "zwei", "drei"], 0.02)
    test_x = map(string, 1:3)
    
    context("constructors") do
        @fact typeof(clfa.cols) => Array{ASCIIString, 1}
        @fact typeof(clfa.alpha) => Float64
        @fact typeof(clfa.ndims) => Int64
        @fact typeof(clfa.w) => Array{Float64, 1}
        
        @fact typeof(clfd.cols) => Array{ASCIIString, 1}
        @fact typeof(clfd.alpha) => Float64
        @fact typeof(clfd.w) => Dict{ASCIIString, Float64}
    end

    context("get_x") do
        xt = ogd.get_x(clfa, test_x)
        @fact [typeof(xi) for xi in xt] => [Int64, Int64, Int64, Int64]
        @fact xt .> 0 => [true, true, true, true]
        
        xt = ogd.get_x(clfd, test_x)
        @fact xt => ["BIAS", "eins__1", "zwei__2", "drei__3"]
    end
    
    context("get_w") do
        @fact ogd.get_w(clfa, [1, 2, 3]) => roughly([0., 0., 0.])
        @fact ogd.get_w(clfa, 1) => roughly(0.)
        
        @fact ogd.get_w(clfd, test_x) => roughly([0., 0., 0.])
        @fact ogd.get_w(clfd, "1") => roughly(0.)
    end
    
    context("get_p") do
        xt = ogd.get_x(clfa, test_x)
        @fact ogd.get_p(clfa, xt) => 0.5
        
        xt = ogd.get_x(clfd, test_x)
        @fact ogd.get_p(clfd, xt) => 0.5
    end
end