# Original matlab code from pmtk3 by Daniel Eaton (danieljameseaton@gmail.com)
# Ported by Levi Boyles
include("ars_defs.jl")
include("logging.jl")
include("probability_utils.jl")


#   ARS - Adaptive Rejection Sampling
#         sample perfectly & efficiently from a univariate log-concave
#         function
#
#   logpdf        a function handle to the log of a log-concave function.
#                  evaluated as: logpdf(x)
#
#   domain      the domain of logpdf. may be unbounded.
#                  ex: [1,inf], [-inf,inf], [-5, 20]
#
#   a,b         two points on the domain of logpdf, a<b. if domain is bounded
#                  then use a=domain[1], b=domain[2]. if domain is
#                  unbounded on the left, the derivative of logpdf for x=<a
#                  must be positive. if domain is unbounded on the right,
#                  the derivative for x>=b must be negative.
#
#                  ex: domain = [1,inf], a=1, b=20 (ensuring that
#                  logpdf'(x>=b)<0
#
#   nSamples    number of samples to draw
#
function ars(logpdf::Function, a::Float64, b::Float64, domain::Vector{Float64}, nSamples::Int)
    debug = false

    if domain[1] >= domain[2]
        error("invalid domain")
    end

    if a>=b || isinf(a) || isinf(b) || a<domain[1] || b>domain[2]
        error("invalid a and b")
    end


    numDerivStep = 1e-3 * (b-a)
    S = [a, a+numDerivStep, b-numDerivStep, b]

    # ensure the derivative there is positive
    if domain[1]==-Inf
        f1 = logpdf(S[1])
        f2 = logpdf(S[2])
        if logpdf(S[2]) <= logpdf(S[1])
            error("derivative at a must be positive, since the domain is unbounded to the left");
        end
    end

    # ensure the derivative there is negative
    if domain[2]==Inf
        f3 = logpdf(S[3])
        f4 = logpdf(S[4])
        if f4 >= f3
                error("derivative at b must be negative, since the domain is unbounded to the right");
        end
    end

    # initialize a mesh on which to create upper & lower hulls
    nInitialMeshPoints = 3;
    S = unique([S[1]; S[2]:(S[3]-S[2])/(nInitialMeshPoints+1):S[3]; S[4]]);
    fS = [logpdf(s) for s in S];

    lowerHull, upperHull = arsComputeHulls(S, fS, domain);

    nSamplesNow = 0;
    iterationNo = 1;

    samples = zeros(nSamples)

    while true

#            if debug
#                    figure(1); clf;
#                    arsPlot(upperHull, lowerHull, domain, S, fS, logpdf);
#            end

            # sample x from Hull
            x = arsSampleUpperHull( upperHull );

            lhVal, uhVal = arsEvalHulls( x, lowerHull, upperHull );

            U = rand();

            meshChanged = 0; # flag to indicate if a new point has been added to the mesh

            # three cases for acception/rejection
            if log(U)<=lhVal-uhVal
                    # accept, u is below lower bound
                    nSamplesNow = nSamplesNow + 1;
                    samples[nSamplesNow] = x;

            elseif log(U)<=logpdf(x )-uhVal
                    # accept, u is between lower bound and f
                    nSamplesNow = nSamplesNow + 1;
                    samples[nSamplesNow] = x;

                    meshChanged = 1;
            else
                    # reject, u is between f and upper bound
                    meshChanged = 1;

            end

            if meshChanged == 1
                    S = sort( [S; x] );
                    fS = [logpdf(s) for s in S];


                    lowerHull, upperHull = arsComputeHulls(S, fS, domain);
            end

            if debug
                    print("iteration $iterationNo, samples collected $nSamplesNow");
            end

            iterationNo = iterationNo + 1;

            if nSamplesNow==nSamples
                    break;
            end

    end

    samples
end

function arsComputeHulls(S, fS, domain)

    # compute lower piecewise-linear hull
    # if the domain of logpdf is unbounded to the left or right, then the lower
    # hull takes on -inf to the left or right of the end points of S

    lowerHull = Array(HullNode, length(S)-1)

    for li=1:length(S)-1
            lowerHull[li] = HullNode()
            lowerHull[li].m = (fS[li+1]-fS[li])/(S[li+1]-S[li]);
            lowerHull[li].b = fS[li] - lowerHull[li].m*S[li];
            lowerHull[li].left = S[li];
            lowerHull[li].right = S[li+1];
    end

    # compute upper piecewise-linear hull

    num_upper_segments = 2*(length(S)-2) + isinf(domain[1]) + isinf(domain[2])
    upperHull = Array(HullNode, num_upper_segments)

    i = 0
    if isinf(domain[1])
            # first line (from -infinity)
            m = (fS[2]-fS[1])/(S[2]-S[1]);
            b = fS[1] - m*S[1];
            #pr = 1/m*( exp(m*S[1]+b) - 0 ); # integrating in from -infinity
            pr = computeSegmentLogProb(-Inf, S[1], m, b)

            i += 1
            upperHull[i] = HullNode()
            upperHull[i].m = m;
            upperHull[i].b = b;
            upperHull[i].pr = pr;
            upperHull[i].left = -Inf;
            upperHull[i].right = S[1];

    end

    # second line
    m = (fS[3]-fS[2])/(S[3]-S[2]);
    b = fS[2] - m*S[2];
    #pr = 1/m * ( exp(m*S[2]+b) - exp(m*S[1]+b) );
    pr = computeSegmentLogProb(S[1], S[2], m, b)


    i += 1
    upperHull[i] = HullNode()
    upperHull[i].m = m;
    upperHull[i].b = b;
    upperHull[i].pr = pr;
    upperHull[i].left = S[1];
    upperHull[i].right = S[2];

    # interior lines
    # there are two lines between each abscissa
    for li=2:length(S)-2

            m1 = (fS[li]-fS[li-1])/(S[li]-S[li-1]);
            b1 = fS[li] - m1*S[li];

            m2 = (fS[li+2]-fS[li+1])/(S[li+2]-S[li+1]);
            b2 = fS[li+1] - m2*S[li+1];

            if !isfinite(m1) && !isfinite(m2)
                lprintln("S: $(S[li-1:li+2])")
                lprintln("fS: $(fS[li-1:li+2])")
            
                error("both hull slopes are infinite")
            end


            #ix = (b1-b2)/(m2-m1); # compute the two lines' intersection

            dx1 = S[li]-S[li-1]
            df1 = fS[li]-fS[li-1]
            dx2 = S[li+2]-S[li+1]
            df2 = fS[li+2]-fS[li+1]
          
            f1 = fS[li]
            f2 = fS[li+1]
            x1 = S[li]
            x2 = S[li+1]
 
            # more numerically stable than above 
            #ix = (b1-b2)*dx1*dx2 / ( df2*dx1 - df1*dx2)
            ix = ((f1*dx1-df1*x1)*dx2 - (f2*dx2-df2*x2)*dx1) / ( df2*dx1 - df1*dx2)

            if !isfinite(m1) || abs(m1-m2) < 10.0^8 * eps(m1)
                ix = S[li]
                pr1 = -Inf
                #pr2 = 1.0/m2 * ( exp(m2*S[li+1]+b2) - exp(m2*ix+b2) );
                pr2 = computeSegmentLogProb(ix, S[li+1], m2, b2)
            elseif !isfinite(m2)
                ix = S[li+1]
                #pr1 = 1.0/m1 * ( exp(m1*ix+b1) - exp(m1*S[li]+b1) );
                pr1 = computeSegmentLogProb(S[li], ix, m1, b1)
                pr2 = -Inf
            else

                if !isfinite(ix)
                    lprintln("b1: $b1, b2: $b2, m1: $m1, m2: $m2")
                    error("Non finite intersection")
                end

                if abs(ix - S[li]) < 10.0^12 *eps(S[li])
                    ix = S[li]
                elseif abs(ix-S[li+1]) < 10.0^12 *eps(S[li+1]) 
                    ix = S[li+1]
                end

                if ix < S[li] || ix > S[li+1]
                    lprintln(fS[li-1:li+2]) 
                    lprintln(S[li-1:li+2])
                    lprintln("b1: $b1, b2: $b2, m1: $m1, m2: $m2")
                    lprintln("ix: $ix")
                    error("Intersection out of bounds -- logpdf is not concave")
                end

#                pr1 = 1.0/m1 * ( exp(m1*ix+b1) - exp(m1*S[li]+b1) );
#                pr2 = 1.0/m2 * ( exp(m2*S[li+1]+b2) - exp(m2*ix+b2) );
                pr1 = computeSegmentLogProb(S[li], ix, m1, b1)
                pr2 = computeSegmentLogProb(ix, S[li+1], m2, b2)
            end
    


            i += 1;
            upperHull[i] = HullNode()
            upperHull[i].m = m1;
            upperHull[i].b = b1;
            upperHull[i].pr = pr1;
            upperHull[i].left = S[li];
            upperHull[i].right = ix;

            i += 1;
            upperHull[i] = HullNode()
            upperHull[i].m = m2;
            upperHull[i].b = b2;
            upperHull[i].pr = pr2;
            upperHull[i].left = ix;
            upperHull[i].right = S[li+1];

    end

    # second last line
    m = (fS[end-1]-fS[end-2])/(S[end-1]-S[end-2]);
    b = fS[end-1] - m*S[end-1];
    #pr = 1.0/m * ( exp(m*S[end]+b) - exp(m*S[end-1]+b) );
    pr = computeSegmentLogProb(S[end-1], S[end], m ,b)

    i += 1
    upperHull[i] = HullNode()
    upperHull[i].m = m;
    upperHull[i].b = b;
    upperHull[i].pr = pr;
    upperHull[i].left = S[end-1];
    upperHull[i].right = S[end];

    if isinf(domain[2])
            # last line (to infinity)
            m = (fS[end]-fS[end-1])/(S[end]-S[end-1]);
            b = fS[end] - m*S[end];
            #pr = 1.0/m * ( 0 - exp(m*S[end]+b) );
            pr = computeSegmentLogProb(S[end], Inf, m, b)


            i += 1 
            upperHull[i] = HullNode()
            upperHull[i].m = m;
            upperHull[i].b = b;
            upperHull[i].pr = pr;
            upperHull[i].left = S[end];
            upperHull[i].right = Inf;
    end

#    Z = sum([upperHull[i].pr for i = 1:length(upperHull)]);
#    #lprintln([upperHull[i].pr for i = 1:length(upperHull)]);
#    for li=1:length(upperHull)
#            upperHull[li].pr = upperHull[li].pr / Z;
#    end

    probs = exp_normalize([upperHull[i].pr for i = 1:length(upperHull)])
    for li=1:length(upperHull)
            upperHull[li].pr = probs[li]
    end

    (lowerHull, upperHull)
end

function computeSegmentLogProb(l, r, m, b)
    #1.0/m * (exp(m*r+b) - exp(m*l+b))
    if l == -Inf
        return -log(m) + m*r+b 
    elseif r == Inf
        return -log(-m) + m*l+b 
    else
        M = max(m*r+b, m*l+b)
        return -log(abs(m)) + log(abs(exp(m*r+b-M) - exp(m*l+b-M))) + M 
    end
end

function arsSampleUpperHull( upperHull )

    cdf = cumsum([upperHull[i].pr for i = 1:length(upperHull)]);

    # randomly choose a line segment
    U = rand();
    li = 0
    for li = 1:length(upperHull)
            if( U<cdf[li] ) break;
            end
    end

    #lprintln(cdf)

    # sample along that line segment
    U = rand();

    m = upperHull[li].m;
    b = upperHull[li].b;
    left = upperHull[li].left;
    right = upperHull[li].right;

    M = max(m*right, m*left)

    #x = (log( U*(exp(m*right+b) - exp(m*left+b)) + exp(m*left+b) ) - b) / m ;
    x = (log( U*(exp(m*right-M) - exp(m*left-M)) + exp(m*left-M) ) + M) / m ;


    @assert x >= left && x <= right
    if isinf(x) || isnan(x)
            lprintln(upperHull[li])
            error("sampled an infinite or NaN x");
    end
    x
end

function arsEvalHulls( x, lowerHull, upperHull )

    lhVal = 0.0
    # lower bound
    if x<minimum([lowerHull[i].left for i = 1:length(lowerHull)])
            lhVal = -Inf;
    elseif x>maximum([lowerHull[i].right for i = 1:length(lowerHull)]);
            lhVal = -Inf;
    else
            for li=1:length(lowerHull)
                    left = lowerHull[li].left;
                    right = lowerHull[li].right;

                    if x>=left && x<=right
                            lhVal = lowerHull[li].m*x + lowerHull[li].b;
                            break;
                    end
            end
    end

    uhVal = 0.0
    # upper bound
    for li = 1:length(upperHull)
            left = upperHull[li].left;
            right = upperHull[li].right;

            if x>=left && x<=right
                    uhVal = upperHull[li].m*x + upperHull[li].b;
                    break;
            end
    end
    (lhVal, uhVal)
end

#function arsPlot(upperHull, lowerHull, domain, S, fS, logpdf)
#
#
#    Swidth = S[end]-S[1];
#    plotStep = Swidth/1000;
#    ext = 0.15*Swidth; # plot this much before a and past b, if the domain is infinite
#
#    left = S[1]; right = S[end];
#    if isinf(domain[1])
#        left = left - ext
#    end
#    if isinf(domain[2])
#        right = right + ext
#    end
#
#    x = left:plotStep:right;
#    fx = logpdf(x);
#
#    plot(x,fx, "k-"); hold on;
#    plot(S, fS, 'ko');
#    title('ARS');
#
#    # plot lower hull
#    for li=1:length(S)-1
#
#            m = lowerHull[li].m;
#            b = lowerHull[li].b;
#
#            x = lowerHull[li].left:plotStep:lowerHull[li].right;
#            plot( x, m*x+b, 'b-' );
#
#    end
#
#    # plot upper bound
#
#    # first line (from -infinity)
#    if isinf(domain[1])
#            x = (upperHull[1].right-ext):plotStep:upperHull[1].right;
#            m = upperHull[1].m;
#            b = upperHull[1].b;
#            plot( x, x*m+b, 'r-');
#    end
#
#    # middle lines
#    for li=2:length(upperHull)-1
#
#            x = upperHull[li].left:plotStep:upperHull[li].right;
#            m = upperHull[li].m;
#            b = upperHull[li].b;
#            plot( x, x*m+b, 'r-');
#
#    end
#
#    # last line (to infinity)
#    if isinf(domain[2])
#            x = upperHull[end].left:plotStep:(upperHull[end].left+ext);
#            m = upperHull[end].m;
#            b = upperHull[end].b;
#            plot( x, x*m+b, 'r-');
#    end
#
