#!/usr/bin/env python

import sys
from math import ceil, erf, exp, pi, sqrt

X_MIN = 1.0e-11  # MeV
X_MAX = 20.0     # MeV

FP_FMT = "%12.6e"  # Format for FP values

class FWatt:
    """Callable that evaluates the antiderivative of the
       Watt fission spectrum c * exp(-x/a) * sinh(sqrt(b*x)) at x:

       F(x) = a/4 ( sqrt(pi*a*b) * exp(a*b/4) *
                        ( erf((2*y - a*b) / (2*sqrt(a*b))) +
                          erf((2*y + a*b) / (2*sqrt(a*b))) )
                    - 2 * (exp(2*y) - 1) * exp(-y*(a*b+y)/(a*b)) )

       where y = sqrt(b*x)
    """
    def __init__(self, a = 0.965, b = 2.29):
        assert a > 0
        assert b > 0
        c1 = a * b
        c2 = sqrt(c1)
        c3 = self.c3 = 0.25 * a * sqrt(pi) * c2 * exp(0.25 * c1)
        c4 = self.c4 = 1.0 / c2
        c5 = self.c5 = 0.5 * c1 * c4
        c6 = self.c6 = 0.5 * a
        c7 = self.c7 = 1.0 / c1
        self.b = b
        # Normalize
        c = 1.0 / (self(X_MAX) - self(X_MIN))
        self.c3 *= c
        self.c6 *= c
    def __call__(self, x):
        assert X_MIN <= x <= X_MAX
        y  = sqrt(self.b * x)
        k1 = self.c4 * y
        k2 = k1 - self.c5
        k3 = k1 + self.c5
        k4 = exp(2.0 * y) - 1.0
        k5 = y * (1.0 + self.c7 * y)
        k6 = exp(-k5)
        return self.c3 * (erf(k2) + erf(k3)) - self.c6 * k4 * k6


def equiprobable(F, x_hi, nbins, tol = 1.0e-12):
    """Find (nbins + 1) equiprobable bin boundaries of the derivative of
       F(x) in the interval [X_MIN, x_hi] with given tolerance. Return the
       bin boundaries and the bin probability.
    """
    assert X_MIN <= x_hi <= X_MAX
    if nbins == 0:
        assert x_hi == X_MIN
        return [], 0.0
    assert nbins > 0
    # Calculate total and bin probability
    p_tot = F(x_hi) - F(X_MIN)
    p_bin = p_tot / nbins
    # Calculate bin boundaries
    x = X_MIN
    dx = (x_hi - X_MIN) / nbins
    bins = [X_MIN]
    for i in xrange(nbins - 1):
        while True:
            p = F(x + dx) - F(x)
            if abs(p - p_bin) <= tol * p_bin:
                break
            dx *= p_bin / p
        x += dx
        assert x < x_hi
        bins.append(x)
    bins.append(x_hi)
    # Return bin boundaries and bin probability
    return bins, p_bin


def uniform(F, x_lo, max_dx):
    """Return a list of uniform-width bin probabilities in the interval
       [x_lo, X_MAX] with maximum width max_dx.
    """
    assert X_MIN <= x_lo <= X_MAX
    if x_lo == X_MAX:
        return []
    assert max_dx > 0.0
    # Calculate number of bins
    nbins = (X_MAX - x_lo) / max_dx
    nbins = int(ceil(nbins - 1.0e-6))
    # Calculate bin boundaries
    dx = (X_MAX - x_lo) / nbins
    bins = [x_lo + dx * i for i in xrange(nbins + 1)]
    # Calculate bin probabilities
    cdf = map(F, bins)
    pdf = [cdf[i+1] - cdf[i] for i in xrange(nbins)]
    # Return bin boundaries and probabilities
    return pdf


def getarg(i, default=None):
    """Return the i-th command line argument. If the index is out of range,
       return the given default value.
    """
    try:
        return sys.argv[i]
    except IndexError:
        return default


def format_card(body, name=""):
    indent = 6
    slist = []
    pos = 0
    # Format the card name
    slist.append("%-*s" % (indent, name))
    pos += indent
    # Format the card body
    for word in body:
        wlen = len(word) + 1
        if pos + wlen > 72:
            slist.append("\n%-*s" % (indent, ""))
            pos = indent
        slist.append(word)
        pos += wlen
    # Return the formatted string
    return " ".join(slist)


if __name__ == "__main__":

    # Set defaults
    dist = None    # SDEF distribution number
    nepb = None    # number of equiprobable bins
    eb   = 6.0     # boundary btw equiprobable and uniform bins [MeV]
    umax = 0.5     # maximum width of uniform bins [MeV]
    a    = 0.965   # Watt constant 1 [MeV]
    b    = 2.29    # Watt constant 2 [1/Mev]

    # Output usage
    argc = len(sys.argv)
    if (argc < 3) or (argc > 7):
        print("usage: %s dist nepb [eb [umax [a [b]]]]" % sys.argv[0])
        print("  dist   SDEF distribution number to output")
        print("  nepb   number of equiprobable bins to generate")
        print("  eb     boundary btw equiprobable and uniform bins",)
        print(         "(default = %.1f MeV)" % eb)
        print("  umax   maximum width of uniform bins",)
        print(         "(default = %.1f MeV)" % umax)
        print("  a      Watt constant 1 (default = %.3f MeV)" % a)
        print("  b      Watt constant 2 (default = %.3f 1/MeV)" % b)
        sys.exit()

    # Parse command line options
    dist =   int(getarg(1))
    nepb =   int(getarg(2))
    eb   = float(getarg(3, eb))
    umax = float(getarg(4, umax))
    a    = float(getarg(5, a))
    b    = float(getarg(6, b))
    if not (X_MIN <= eb <= X_MAX):
        print("Invalid energy boundary (eb): %.6e," %eb,)
        print("must be between %.3e and %.3f MeV" % (X_MIN, X_MAX))
        sys.exit()
    if (nepb == 0) and (eb != X_MIN):
        print("Number of equiprobable bins must be > 0",)
        print("unless eb = %.3e" % X_MIN)
        sys.exit()

    # Construct indefinite integral
    F = FWatt(a, b)

    # Construct bins and probabilities
    epb, ep_prob = equiprobable(F, eb, nepb)
    u_probs = uniform(F, eb, umax)
    nub = len(u_probs)  # number of uniform bins

    # Build comment cards
    com1 = "c      Watt spectrum (a = %.5f, b = %.5f)" % (a, b)
    com2 = "c        from %.3e to %.1f MeV" % (X_MIN, X_MAX)
    com3 = "c        with %4d equiprobable bins below %.3f MeV" % (nepb, eb)
    com4 = "c        with %4d      uniform bins above %.3f MeV" % (nub, eb)

    # Build SI card
    si_body = []
    for x in epb:
        si_body.append(FP_FMT % x)
    if nub:
        if not si_body:
            si_body = [FP_FMT % X_MIN]
        si_body.append("%di" % (nub - 1))
        si_body.append(FP_FMT % X_MAX)

    # Build SP card
    sp_body = []
    if nepb:
        sp_body = ["0", FP_FMT % ep_prob, "%dr" % (nepb - 1)]
    for p in u_probs:
        sp_body.append(FP_FMT % p)

    # Build SB card
    nbins = nepb + nub
    sb_body = ["0", FP_FMT % (1.0 / nbins), "%dr" % (nbins - 1)]

    # print cards
    print(com1)
    print(com2)
    if nepb:
        print(com3)
    if nub:
        print(com4)
    print(format_card(si_body, "si%-4d" % dist))
    print(format_card(sp_body, "sp%-4d" % dist))
    print(format_card(sb_body, "sb%-4d" % dist))
