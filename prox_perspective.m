function [b, a] = prox_perspective(bt, at, c)
%PROX_PERSPECTIVE  Prox of the perspective function  c * a^2 / (2*b).  (v3)
%
%   [b,a] = PROX_PERSPECTIVE(bt, at, c) solves, elementwise,
%
%       (b,a) = argmin_{b,a}  (b-bt)^2/2 + (a-at)^2/2 + c*a^2/(2*b)
%
%   with the convention  a^2/(2b) = 0 if a=0,b=0;  +Inf if b<=0, a~=0.
%
%   Reduction (cf. Papadakis-Peyre-Oudet, SIIMS 2014):
%       a = b*at/(b+c),   b = largest real root of
%       g(b) = (b - bt)*(b + c)^2 - (c/2)*at^2 = 0,
%   and (b,a) = (0,0) if that root is <= 0.
%
%   v2: closed-form start. With x = b + c the cubic is
%       x^3 - p x^2 - r = 0,   p = bt + c,  r = (c/2) at^2 >= 0,
%   whose largest real root is computed by Cardano (one real root) or the
%   trigonometric formula (three real roots). The estimate is nudged up and
%   polished by MONOTONE Newton from above (g is increasing right of the
%   largest root), so the v1 convergence guarantee is kept; elements where
%   the nudged start is not above the root fall back to the v1 start
%       b0 = max(bt,0) + r/(max(bt,0)+c)^2   (g(b0) >= 0 proven).
%   Typically 1-3 Newton steps instead of up to 60.
%
%   c may be scalar, implicitly expanding, or full size; all c > 0.
%   Checked against brute force in test_hk_unit.m (T1).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

c = c + zeros(size(bt));
if any(c(:) <= 0)
    error('prox_perspective: all weights c must be > 0');
end

p = bt + c;
r = 0.5 * c .* at.^2;

% ---- closed-form largest root x of x^3 - p x^2 - r = 0 -------------
x  = zeros(size(p));
A3 = p.^3 / 27;
Dl = r .* (A3 + r/4);                    % discriminant (>= 0: one real root)
one = Dl >= 0;
if any(one(:))
    % numerically stable Cardano (v3): u = cbrt(Ah + sign(Ah)*s) has no
    % cancellation; the second cube root is P/(3u) with P = -p^2/3, i.e.
    % y = u + p^2/(9u). (v2 used cbrt(Ah+s) + cbrt(Ah-s): cancellation
    % for |at| >> c gave a start BELOW the root.)
    s  = sqrt(Dl(one));
    Ah = A3(one) + r(one)/2;
    po = p(one);
    sg = 2*(Ah >= 0) - 1;
    uu = cbrt(Ah + sg.*s);
    y  = zeros(size(uu));
    nz = uu ~= 0;
    y(nz) = uu(nz) + po(nz).^2 ./ (9*uu(nz));
    x(one) = y + po/3;
end
three = ~one;                            % only possible when p < 0
if any(three(:))
    ap  = abs(p(three));
    arg = min(max(-1 + 27*r(three) ./ (2*ap.^3), -1), 1);
    x(three) = (2*ap/3) .* cos(acos(arg)/3) + p(three)/3;
end
b = x - c;

% ---- push just above the root (g >= 0), geometrically if needed ----
% (v3: v2 fell back to the far v1 start here and could run out of Newton
% steps). Only non-finite estimates use the proven v1 start.
scl = abs(b) + c + abs(bt);
b   = b + 1e-12 * scl;
for k = 0:59
    g  = (b - bt).*(b + c).^2 - r;
    lo = ~(g >= 0) & isfinite(b);
    if ~any(lo(:)); break; end
    b(lo) = b(lo) + (2^k) * 1e-12 * scl(lo);
end
bad = ~isfinite(b);
if any(bad(:))
    btp    = max(bt(bad), 0);
    b(bad) = btp + r(bad) ./ (btp + c(bad)).^2;
end

% ---- monotone Newton polish ----------------------------------------
for it = 1:100
    bc   = b + c;
    g    = (b - bt).*bc.^2 - r;
    gp   = bc.^2 + 2*(b - bt).*bc;
    step = g ./ max(gp, realmin);
    b    = b - step;
    % PER-ELEMENT relative stop (a global test would let small-scale
    % elements stop unconverged when scales differ across the array)
    if all(abs(step(:)) <= 1e-13 * (abs(b(:)) + c(:) + abs(bt(:))))
        break
    end
end

neg    = (b <= 0) | ~isfinite(b);
b(neg) = 0;
a      = b .* at ./ (b + c);
a(neg) = 0;
end

% ======================================================================
function y = cbrt(z)
y = sign(z) .* abs(z).^(1/3);
end
