%RUN_E1_HK  E1: HK-dEIT protocol P1 on the healthy-volunteer data  (v2)
%   (Gomez-Laberge 2005, 1-plane and 2-plane 8+8 zig-zag).
%
%   Prerequisite: prep_E1.m -> E1_prep.mat (QC, 13 fps verified, cardiac
%   low-pass, oriented models; orientation verified by the cardiac map).
%
%   Recordings: _001 tidal breathing (control: gas enters/leaves through the
%   airway -> global source dominates) and _003 paradoxical breathing (breath
%   held, gas shifted rib cage <-> abdomen, ~10 s cycle).
%   E1 prediction (b): in the 2-plane model the paradoxical shift is largely
%   cranio-caudal TRANSPORT; the 1-plane model cannot see the out-of-plane
%   exchange and shows it as a global change instead.
%   CONFOUND: chest-wall / abdominal motion changes the boundary shape and the
%   electrode positions, which EIT reads as a global impedance change even at
%   constant gas volume; no method can separate this from gas.
%
%   v2 (after v1): (1) the auxiliary GN hyperparameter is chosen by the
%   L-CURVE corner (no noise model needed) - GCV picked its lower bound on
%   the 1-plane data (under-regularized, noise-dominated GN images). The
%   discrepancy-principle value (noise from the > 2 Hz raw content, scaled
%   to the retained cardiac-low-pass band) is printed for comparison only:
%   in vivo, model errors exceed the measurement noise, so it would again
%   under-regularize; (2) the ventilation profile w and rho0 come from the TIDAL
%   record _001 of the same model (w = TIV map) and are used for both _001
%   and _003 - in v1 w was the temporal std of the paradoxical record itself,
%   which is > 0 in BOTH anti-phase regions and let the free mode absorb most
%   of the shift; (3) PV is compared as the peak-to-peak swing of the region
%   content (HK and GN) plus the HK flux-based value; (4) the amplitude of
%   the ventilation mode is reported (should be much smaller in _003).
%
%   ~30 min. Needs EIDORS.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
S = load(fullfile(here, 'E1_prep.mat'));
recs = S.recs_save;  fps = S.fps;

% ---------------- knobs ----------------
ids     = {'001', '003'};
win_s   = 20;   nwin = 2;   ds = 2;
P1 = struct('alpha', 1e-3, 'beta', 1e-2, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 4000;
f_noise = 2.0;                   % Hz: content above this is taken as noise
dt = ds / fps;

models = {S.fmdl1, S.fmdl2};
R = struct('plane', {}, 'id', {}, 'win', {}, 'iface', {}, 'Rg_hk', {}, 'Rg_gn', {}, ...
           'Tshare', {}, 'Vshare', {}, 'sw_hk', {}, 'sw_gn', {}, 'PVflux', {}, 'hp', {});
for pl = 1:2
    fm  = models{pl};
    Ne  = size(fm.elems, 1);
    lung_idx = [fm.mat_idx{2}(:); fm.mat_idx{3}(:)];
    s0 = ones(Ne, 1);  s0(lung_idx) = sig_lung;
    J  = calc_jacobian(mk_image(fm, s0));                  % normalized (fm is)
    grp = zeros(Ne,1);  grp(fm.mat_idx{2}) = 1;  grp(fm.mat_idx{3}) = 2;
    Gc = calc_dual_graph(fm, lung_idx, struct('target_N', N_coarse, 'groups', grp));
    Ac = J * Gc.EL;
    yl  = Gc.cent(:,2);
    ifc = struct('name', {}, 'r1', {}, 'L', {});
    ifc(end+1) = struct('name', 'anterior|posterior', 'r1', yl > wmedian(yl, Gc.vol), 'L', NaN);
    if pl == 2
        ifc(end+1) = struct('name', 'cranial|caudal', ...
                            'r1', (Gc.cent(:,3) - 0.5) * sign(S.sup_z) > 0, 'L', NaN);
    end
    for f = 1:numel(ifc)
        r1 = ifc(f).r1;
        c1 = sum(Gc.cent(r1,:) .* Gc.vol(r1), 1) / sum(Gc.vol(r1));
        c2 = sum(Gc.cent(~r1,:) .* Gc.vol(~r1), 1) / sum(Gc.vol(~r1));
        ifc(f).L = norm(c1 - c2);
    end
    main = numel(ifc);

    % ---- ventilation profile and rho0 from the TIDAL record of this model ----
    rt  = recs([recs.plane] == pl & strcmp({recs.id}, '001'));
    frt = 1:ds:size(rt.V_lp, 2);
    Yt  = calc_difference_data(rt.vh_low, rt.V_lp(:, frt), fm);
    hpt = lcurve_hp(Ac, Yt);
    rgt = -gn_frames(Ac, Yt, hpt);
    gt  = Gc.vol' * rgt;  [~, o] = sort(gt);  n10 = max(3, round(0.1*numel(gt)));
    w_tidal = max(mean(rgt(:, o(end-n10+1:end)), 2) - mean(rgt(:, o(1:n10)), 2), 0);
    rho0    = P1.kappa * sum(Gc.vol .* w_tidal) / sum(Gc.vol);
    fprintf('\n######## %d-plane: Nc %d | interfaces %s (L %s) | tidal TIV map from 001 (hp %.2g), rho0 %.3g\n', ...
            pl, Gc.N, strjoin({ifc.name}, ', '), mat2str([ifc.L], 3), hpt, rho0);

    for ii = 1:numel(ids)
        r = recs([recs.plane] == pl & strcmp({recs.id}, ids{ii}));
        if strcmp(r.kind, 'paradox'); vh = r.vh_mean; else; vh = r.vh_low; end
        sig2 = noise_var(r, fm, fps, f_noise);
        nfr  = round(win_s * fps);
        for w = 1:nwin
            fr = (w-1)*nfr + (1:ds:nfr);
            if fr(end) > size(r.V_lp, 2); break; end
            Yd  = calc_difference_data(vh, r.V_lp(:, fr), fm);
            [hp, hedge] = lcurve_hp(Ac, Yd);                    % primary: L-curve
            hpd = discrepancy_hp(Ac, Yd, sig2);                 % reported for comparison
            if hedge; warning('E1: L-curve corner at the edge of the lambda grid'); end
            rgn = -gn_frames(Ac, Yd, hp);                       % GN (rho = -dsigma)

            o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*ifc(main).L/pi, ...
                       'gamma', 1e3, 'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, ...
                       'use_eta', false, 'normalize', true, 'tau0', tau0, ...
                       'rho0', rho0, 'vent_w', w_tidal);
            [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yd, Gc, o);
            gsrc = dt * (Gc.vol' * info.zeta);                 % global source per step
            gV   = Gc.vol' * rho;
            Vv   = [0, cumsum(dt * info.vent_c)];               % volume explained by the mode
            Vsh  = (max(Vv) - min(Vv)) / max(max(gV) - min(gV), eps);
            for f = 1:numel(ifc)
                r1  = ifc(f).r1;
                QG  = hk_pendelluft(Gc, m, r1, dt);            % flux RATE into region 1
                Rgh = rg_metric(rho, Gc.vol, r1);
                Rgg = rg_metric(rgn, Gc.vol, r1);
                Tsh = sum(abs(dt*QG)) / (sum(abs(dt*QG)) + sum(abs(gsrc)) + eps);
                swh = p2p(Gc.vol(r1)' * rho(r1,:));
                swg = p2p(Gc.vol(r1)' * rgn(r1,:));
                PVf = dt * sum(max(QG, 0));                    % per window
                R(end+1) = struct('plane', pl, 'id', ids{ii}, 'win', w, 'iface', ifc(f).name, ...
                                  'Rg_hk', Rgh, 'Rg_gn', Rgg, 'Tshare', Tsh, 'Vshare', Vsh, ...
                                  'sw_hk', swh, 'sw_gn', swg, 'PVflux', PVf, 'hp', hp); %#ok<SAGROW>
                fprintf(['%dpl %s (%s) win %d | %-18s | Rg HK %.3f GN %.3f | T-share %.2f | ' ...
                         'vent-mode share %.2f | swing HK %.3e GN %.3e | flux-PV %.3e | hp L %.2g (discr %.2g) res %.0e\n'], ...
                         pl, ids{ii}, r.kind, w, ifc(f).name, Rgh, Rgg, Tsh, Vsh, swh, swg, PVf, ...
                         hp, hpd, max(info.res));
            end
            if w == 1
                E1fig.(sprintf('p%d_%s', pl, ids{ii})) = struct('rho', rho, 'rgn', rgn, ...
                    'gsrc', gsrc, 'm', m, 'vol', Gc.vol, 'ifc', ifc, 'dt', dt, ...
                    'Vv', Vv, 'Gc', Gc);
            end
        end
    end
end
save(fullfile(here, 'E1_results.mat'), 'R', 'E1fig', 'P1', 'win_s', 'ds', 'f_noise');

% ---------------- summary ----------------
fprintf('\n================ E1 SUMMARY v2 (mean over windows) ================\n');
fprintf('%-5s %-4s %-18s %7s %7s %7s %9s %10s %10s %10s\n', 'model', 'rec', 'interface', ...
        'Rg HK', 'Rg GN', 'T-shr', 'vent-shr', 'swing HK', 'swing GN', 'flux-PV');
tag = cellfun(@(a,b,c) sprintf('%d|%s|%s', a, b, c), {R.plane}, {R.id}, {R.iface}, 'UniformOutput', false);
keys = unique(tag, 'stable');
for k = 1:numel(keys)
    Rk = R(strcmp(tag, keys{k}));
    fprintf('%-5s %-4s %-18s %7.3f %7.3f %7.2f %9.2f %10.3e %10.3e %10.3e\n', ...
            sprintf('%dpl', Rk(1).plane), Rk(1).id, Rk(1).iface, mean([Rk.Rg_hk]), ...
            mean([Rk.Rg_gn]), mean([Rk.Tshare]), mean([Rk.Vshare]), mean([Rk.sw_hk]), ...
            mean([Rk.sw_gn]), mean([Rk.PVflux]));
end
fprintf(['\nExpected: 003 vs 001 -> vent-mode share much smaller (glottis closed), T-share larger;\n' ...
         '2pl/003 cranial|caudal -> smaller Rg than 1pl/003; HK swing ~ GN swing (same data).\n']);

% ---------------- figure ----------------
figure('Name', 'E1 v2: paradoxical breathing (window 1)');
pn = {'p1_003', 'p2_003'};
for q = 1:2
    if ~isfield(E1fig, pn{q}); continue; end
    F  = E1fig.(pn{q});  t = (0:size(F.rho,2)-1) * F.dt;
    f  = numel(F.ifc);  r1 = F.ifc(f).r1;
    subplot(2, 2, q);
    plot(t, F.vol' * F.rho, 'k', t, F.vol(r1)' * F.rho(r1,:), 'b', ...
         t, F.vol(~r1)' * F.rho(~r1,:), 'r', t, F.Vv, 'k:');
    legend('global', F.ifc(f).name, 'other side', 'vent mode', 'Location', 'best');
    title(sprintf('%d-plane: HK gas content', q));  xlabel('t [s]');
    subplot(2, 2, 2 + q);
    QG = hk_pendelluft(F.Gc, F.m, r1, F.dt);
    plot(t(1:end-1), QG, 'b', t(1:end-1), F.gsrc / F.dt, 'k');
    legend(sprintf('flux into %s', F.ifc(f).name), 'global source', 'Location', 'best');
    title(sprintf('%d-plane: HK transport vs source (rates)', q));  xlabel('t [s]');
end

% ======================================================================
function Rg = rg_metric(rho, vol, r1)
gV = vol' * rho;
sw = (vol(r1)' * rho(r1,:) - vol(~r1)' * rho(~r1,:)) / 2;
Rg = p2p(gV) / max(p2p(sw), eps);
end

function v = p2p(x); v = max(x) - min(x); end

function s2 = noise_var(r, fm, fps, f_noise)
% per-channel noise variance of the NORMALIZED difference data AS USED
% (cardiac low-pass at r.fc): white-noise density from the raw content above
% f_noise (above the cardiac band), times the retained bandwidth r.fc
V  = r.V;  T = size(V, 2);
f  = abs([0:floor(T/2), -ceil(T/2)+1:-1]) * fps / T;
Vh = real(ifft(fft(V - mean(V, 2), [], 2) .* double(f > f_noise), [], 2));
s2ch = var(Vh, 0, 2) / max(fps/2 - f_noise, eps) * r.fc;     % density x bandwidth
vh   = mean(V, 2);
if isfield(fm, 'normalize_measurements') && fm.normalize_measurements
    s2ch = s2ch ./ vh.^2;
end
s2 = mean(s2ch);
end

function hp = discrepancy_hp(A, Y, s2)
% NOSER-Wiener estimator; lambda such that the mean residual per frame
% equals M * s2 (discrepancy principle); bisection in log(lambda)
p = 1 ./ sqrt(sum(A.^2, 1))';
S = A * (p .* A');  S = (S + S')/2;
[U, D] = eig(S);  s = max(diag(D), 0);
Z = U' * Y;  M = size(A, 1);  T = size(Y, 2);
res = @(lam) sum(sum(((lam ./ (s + lam)) .* Z).^2)) / T;
target = M * s2;
lo = 1e-10 * mean(s);  hi = 1e4 * mean(s);
if res(lo) >= target; hp = sqrt(lo); return; end
if res(hi) <= target; hp = sqrt(hi); return; end
for it = 1:80
    mid = sqrt(lo * hi);
    if res(mid) > target; hi = mid; else; lo = mid; end
end
hp = sqrt(sqrt(lo * hi));
end

function [hp, edge] = lcurve_hp(A, Y)
% L-curve corner (max curvature of log||residual|| vs log||x||_NOSER) for
% the NOSER-Wiener estimator, pooled over frames. In the eigenbasis of
% S = A P A' (P = NOSER^-1): residual^2 = sum (lam/(s+lam))^2 z^2,
% ||x||^2_{P^-1} = sum s z^2/(s+lam)^2.
p = 1 ./ sqrt(sum(A.^2, 1))';
S = A * (p .* A');  S = (S + S')/2;
[U, D] = eig(S);  s = max(diag(D), 0);
Z2 = sum((U' * Y).^2, 2);
lams = logspace(-8, 2, 101) * mean(s);
rr = zeros(size(lams));  xx = zeros(size(lams));
for i = 1:numel(lams)
    rr(i) = sum((lams(i) ./ (s + lams(i))).^2 .* Z2);
    xx(i) = sum(s .* Z2 ./ (s + lams(i)).^2);
end
x = 0.5*log(rr);  y = 0.5*log(xx);  t = log(lams);
x1 = gradient(x, t);  y1 = gradient(y, t);  x2 = gradient(x1, t);  y2 = gradient(y1, t);
kap = (x1.*y2 - x2.*y1) ./ max((x1.^2 + y1.^2).^1.5, eps);
kap([1:3, end-2:end]) = -Inf;                     % ignore the ends
[~, k] = max(kap);  hp = sqrt(lams(k));
edge = k <= 4 || k >= numel(lams) - 3;
end

function m = wmedian(x, w)
[xs, o] = sort(x);  cw = cumsum(w(o)) / sum(w);
m = xs(find(cw >= 0.5, 1));
end

function X = gn_frames(A, Y, hp)
p   = 1 ./ sqrt(sum(A.^2, 1))';
PAt = p .* A';
X   = PAt * ((A*PAt + hp^2*eye(size(A,1))) \ Y);
end
