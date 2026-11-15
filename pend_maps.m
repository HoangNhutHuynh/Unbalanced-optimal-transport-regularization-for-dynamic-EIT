function MP = pend_maps(Gc, rgn, rho, m, kmin, kei, ant_c, dt)
%PEND_MAPS  Per-cluster maps and regional curves of one recording window, for
%   the per-dataset gallery (make_figs_HK Fig16). Same breaths, region split
%   and quadrature as the runs.
%
%   MP = pend_maps(Gc, rgn, rho, m, kmin, kei, ant_c, dt)
%   rgn, rho  N x T cluster gas (GN, HK; gas increase > 0);  m HK flux;
%   kmin (nb+1) / kei (nb) end-expiration / end-inspiration frames;
%   ant_c anterior clusters;  dt frame interval (s)
%   MP fields
%     tivGN, tivHK  mean tidal gas per unit volume (end-insp - end-exp), / its domain mean
%     oopGN, oopHK  out-of-phase density: mean over breaths of sum_insp |dx_i - a_i dG|
%                   per unit volume, a_i = <dG,dx_i>/<dG,dG> (i.e. the quadrature
%                   readout at cluster level), in % of the mean TV per unit domain
%                   volume -> its volume-weighted mean is a pixel-level %TV
%     VA/VP (HK), VAg/VPg (GN)  regional gas / median TV, whole window
%     r  HK quadrature flux into ant_c / median TV per s;  t, kmin, kei, dt
%     PVq  HK quadrature PV per breath, both directions, in % of the GN tidal volume
%          of that breath (same normalisation as insp_pv(r)./TVg in the runs)
%     TVg, TVh  per-breath tidal volume, GN and HK (model units)
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

kmin = kmin(:)';  kei = kei(:)';  nb = numel(kei);
vol = Gc.vol(:);  Vtot = sum(vol);
[tivGN, oopGN, TVg] = maps_of(rgn);
[tivHK, oopHK, TVh] = maps_of(rho);
QG = hk_pendelluft(Gc, m, ant_c, dt);
F  = (diff(vol' * rho) / dt)';
r  = QG - (F' * QG) / (F' * F) * F;
PVq = nan(nb, 1);
for b = 1:nb
    st = kmin(b):kei(b)-1;
    PVq(b) = 100 * dt * sum(abs(r(st))) / TVg(b);   % GN TV, as the runs (was HK TV: titles did not match the tables)
end
TVm = median(TVh);  TVmg = median(TVg);
MP = struct('tivGN', tivGN, 'tivHK', tivHK, 'oopGN', oopGN, 'oopHK', oopHK, ...
            'VA', vol(ant_c)' * rho(ant_c,:) / TVm, 'VP', vol(~ant_c)' * rho(~ant_c,:) / TVm, ...
            'VAg', vol(ant_c)' * rgn(ant_c,:) / TVmg, 'VPg', vol(~ant_c)' * rgn(~ant_c,:) / TVmg, ...
            'r', r' / TVm, 't', (0:size(rho,2)-1) * dt, 'kmin', kmin, 'kei', kei, 'dt', dt, 'PVq', PVq, 'TVg', TVg, 'TVh', TVh);

    function [tiv, oop, TV] = maps_of(X)
        G  = vol' * X;  TV = arrayfun(@(b) G(kei(b)) - G(kmin(b)), (1:nb)');
        tv = mean(X(:, kei) - X(:, kmin(1:nb)), 2);
        tiv = tv / (sum(vol .* tv) / Vtot);
        dX = diff(X, 1, 2);  dG = diff(G);
        R  = dX - (dX * dG') / (dG * dG') * dG;
        s  = zeros(size(X,1), 1);
        sr = 0;
        for bb = 1:nb
            s  = s + sum(abs(R(:, kmin(bb):kei(bb)-1)), 2);
            sr = sr + sum(abs(vol(ant_c)' * R(ant_c, kmin(bb):kei(bb)-1)));
        end
        % self-check (triangle inequality): cluster-level out-of-phase volume >= the regional (quadGN) one
        assert(sum(vol .* s) >= sr * (1 - 1e-9), 'pend_maps: cluster quadrature below the regional one');
        oop = 100 * (s / nb) / (mean(TV) / Vtot);
    end
end
