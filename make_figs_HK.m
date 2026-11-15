%MAKE_FIGS_HK  Figures of the current HK-dEIT results (thorax FEM, not a disc).
%
%   Reads only saved results (no reconstruction is re-run):
%     E2_results<sfx>.mat   neonate, HK P1 + GN (images, flux, time courses)
%     E2b/E2e/E2f/E2g_v2/E2h_results<sfx>.mat   evaluation on the real background
%       MASK below: 'thorax' (sfx '_thorax', after E2i) or 'lungs' (sfx '', first runs)
%     E3_results/*_thorax_v2.mat + post_timing_shift.mat   paediatric ARDS, patient 1
%     E5b_results/*.mat   piglets
%     E4c/E4d_results.mat   adult thorax simulation (dose-response)
%   Rebuilds the neonate thorax model (mk_library_model 'neonate_16el_lungs')
%   and the cluster graph (domain_graph) exactly as run_E2_neonate, and checks that the
%   graph matches the saved result.
%
%   Display convention: radiological view at the electrode plane - subject's
%   RIGHT on the image LEFT, ANTERIOR up. The neonate was PRONE, so anterior =
%   gravity-dependent. Interface A|P (dashed) = the split used for the flux.
%
%   Output: figs/*.png (300 dpi), figs/*.pdf (vector), figs/*.fig
%     Fig01_geometry        thorax FEM slice, lungs, electrodes, interface
%     Fig02_images          GN vs HK gas images in one breath + TIV maps
%     Fig03_flux            HK transport field, early / mid / late inspiration
%     Fig04_timecourses     global, regional, interface flux (raw, quadrature)
%     Fig05_dose_response   injected pendelluft on the real background
%     Fig06_floors          false-positive floors, raw vs quadrature
%     Fig07_per_breath      real breaths vs pooled null breaths
%     Fig08_asynchrony      quadrature PV and asymmetry vs imposed asynchrony
%     Fig09_timing          timing signature (thirds of inspiration)
%     Fig10_E4d             adult simulation, no inverse crime
%     Fig11_E4c             adult simulation, inverse crime (reference)
%     Fig12_E5b             piglets, PEEP-level-local timing contrast and PV vs calibrated nulls
%     Fig13_E3              paediatric ARDS patient 1, timing contrast and PV vs PEEP
%     Fig14_E6              mechanism simulation: estimators vs ground truth, lag vs true lag
%     Fig15_E7              healthy adults with breath holds: gas trace, tidal vs hold vs null, frames per breath
%     Fig16_gallery         one window per real dataset: TIV (GN, HK), HK out-of-phase density, regional curves
%
%   ~2 min. Needs EIDORS (model + slice mesher) and the files above.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; close all; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
out  = fullfile(here, 'figs');  if ~exist(out, 'dir'); mkdir(out); end

% ---------------- style ----------------
pal  = [0 114 178; 213 94 0; 0 158 115; 230 159 0; 204 121 167] / 255;   % Okabe-Ito (validated)
gry  = [0.55 0.55 0.55];  ink = [0.15 0.15 0.15];
set(groot, 'defaultAxesFontName', 'Helvetica', 'defaultTextFontName', 'Helvetica', ...
    'defaultAxesFontSize', 8, 'defaultTextFontSize', 8, 'defaultAxesTickDir', 'out', ...
    'defaultAxesBox', 'off', 'defaultLineLineWidth', 1.2, 'defaultAxesXColor', ink, ...
    'defaultAxesYColor', ink, 'defaultAxesLineWidth', 0.6);
cmap = diverging_map(256);
W1 = 3.5;  W2 = 7.16;                          % IEEE single / double column [in]

% ---------------- neonate thorax model and cluster graph (as run_E2_neonate) ----------------
fn = fullfile(here, 'Data', 'baby', 'if-neonate-spontaneous (1)', 'P04P-1016.get');
D  = load_real_eit(fn);  fps = D.fps;  dt = 1/fps;
fm = mk_library_model('neonate_16el_lungs');
fm.electrode = fm.electrode(electrode_order(fm, 'left'));
fm.stimulation = D.stim;  fm = mdl_normalize(fm, 1);
Ne = size(fm.elems, 1);
lung_idx = [fm.mat_idx{2}(:); fm.mat_idx{3}(:)];
MASK = 'thorax';   % must match the run_E2* scripts that produced the results
grp = 3 * ones(Ne,1);  grp(fm.mat_idx{2}) = 1;  grp(fm.mat_idx{3}) = 2;
[Gc, sfx, msk] = domain_graph(fm, grp, 800, MASK);  dom_idx = find(msk);
ldf = @(nm) load(fullfile(here, [nm sfx '.mat']));
S2 = ldf('E2_results');
assert(size(S2.rho, 1) == Gc.N, 'cluster graph (%d) differs from the E2 run (%d)', Gc.N, size(S2.rho,1));
rho = S2.rho;  rgn = S2.rgn;  m = S2.m;  kmin = S2.kmin(:)';  kei = S2.kei(:)';  ant_c = logical(S2.ant_c(:));
nb = numel(kei);  T = size(rho, 2);  t = (0:T-1) * dt;
[ant, rgt, c_t] = axes_dir(fm);
H   = max(fm.nodes(:,3)) - min(fm.nodes(:,3));
zel = mean(cell2mat(arrayfun(@(e) fm.nodes(e.nodes,3), fm.electrode(:), 'UniformOutput', false)));
xf  = @(P) [-(P(:,1:2) - c_t) * rgt', (P(:,1:2) - c_t) * ant'];     % radiological view
SL  = slice_geom(fm, zel + 1e-4*H, xf);
isL = false(Ne,1);  isL(fm.mat_idx{2}) = true;  isR = false(Ne,1);  isR(fm.mat_idx{3}) = true;
SL.lung_edges = boundary_edges(SL.tri(ismember(SL.eidx, lung_idx), :));
ep  = zeros(numel(fm.electrode), 2);
for k = 1:numel(fm.electrode); ep(k,:) = mean(fm.nodes(fm.electrode(k).nodes, 1:2), 1); end
ep  = xf(ep);
Rx  = max(abs(SL.XY(:)));
gV  = Gc.vol' * rgn;  hV = Gc.vol' * rho;
TVg = S2.M(:,1);  TVm = median(TVg);
fprintf('model: %d elements, slice %d triangles, %d clusters | median TV %.4g\n', Ne, size(SL.tri,1), Gc.N, TVm);

% ======================= Fig 1: geometry =======================
f1 = newfig(W1, 3.2);
ax = axes(f1);  hold(ax, 'on');
patch(ax, 'Faces', SL.tri, 'Vertices', SL.XY, 'FaceColor', [0.93 0.93 0.93], 'EdgeColor', 'none');
patch(ax, 'Faces', SL.tri(ismember(SL.eidx, find(isL)), :), 'Vertices', SL.XY, 'FaceColor', [0.78 0.87 0.95], 'EdgeColor', 'none');
patch(ax, 'Faces', SL.tri(ismember(SL.eidx, find(isR)), :), 'Vertices', SL.XY, 'FaceColor', [0.80 0.92 0.86], 'EdgeColor', 'none');
draw_edges(ax, SL.XY, SL.bnd_edges, ink, 0.9);
draw_edges(ax, SL.XY, SL.lung_edges, gry, 0.6);
plot(ax, [-1 1]*Rx, [0 0], '--', 'Color', ink, 'LineWidth', 0.8);
plot(ax, ep(:,1), ep(:,2), 's', 'MarkerSize', 5, 'MarkerFaceColor', ink, 'MarkerEdgeColor', 'w');
for k = 1:size(ep,1)
    p = ep(k,:) * 1.13;  text(ax, p(1), p(2), sprintf('%d', k), 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', ink);
end
cL = xf(mean(interp_mesh(fm, 0) .* isL, 1) * Ne / nnz(isL));  cR = xf(mean(interp_mesh(fm, 0) .* isR, 1) * Ne / nnz(isR));
text(ax, cL(1), cL(2), 'L', 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', ink);
text(ax, cR(1), cR(2), 'R', 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Color', ink);
text(ax, 0, 0.05*Rx, 'A|P', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 7, 'Color', ink);
text(ax, 0, 1.28*max(SL.XY(:,2)), 'anterior (dependent, prone)', 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', ink);
axis(ax, 'equal', 'off');  xlim(ax, [-1.3 1.3]*Rx);  ylim(ax, [-1.3 1.4]*Rx);
title(ax, 'Neonate thorax FEM, electrode plane', 'FontWeight', 'normal');
save_fig(f1, fullfile(out, 'Fig01_geometry'), 'image');

% ======================= Fig 2: images =======================
% element-level GN over the whole domain needs the data: rebuild Y as run_E2_neonate
V  = D.vv;  g = sum(V ./ mean(V, 2) - 1, 1);
[S_, fr] = spectrum(g, fps);
RR = peak_in(S_, fr, [0.4 1.6]);  HR = peak_in(S_, fr, [1.7 3.5]);
fc = min(0.8*HR, max(2.2*RR, 0.5*(2*RR + 0.8*HR)));
V_lp = fft_lowpass(V, fps, fc, 0.2*fc);
Y  = calc_difference_data(mean(V_lp(:, kmin), 2), V_lp, fm);
s0 = ones(Ne,1);  s0(lung_idx) = 0.4;
J  = calc_jacobian(mk_image(fm, s0));
hpe = lcurve_hp(J, Y);
xel = -gn_frames(J, Y, hpe);                              % element GN, gas = -dsigma
[~, b0] = min(abs(TVg - median(TVg)));                   % representative breath
k0 = kmin(b0);  k2 = kei(b0);  k1 = round((k0 + k2)/2);
vol = get_elem_volume(fm);
tivE = mean(xel(:, kei) - xel(:, kmin(1:nb)), 2);
tivG = mean(rgn(:, kei) - rgn(:, kmin(1:nb)), 2);   tivH = mean(rho(:, kei) - rho(:, kmin(1:nb)), 2);
tlE  = sum(vol(dom_idx) .* tivE(dom_idx)) / sum(vol(dom_idx));      % mean TIV per volume over the domain
tlG  = sum(Gc.vol .* tivG) / sum(Gc.vol);
rowV = {@(k) xel(:,k) / tlE, @(k) cluster2elem(Gc, rgn(:,k), Ne, dom_idx) / tlG, @(k) cluster2elem(Gc, rho(:,k), Ne, dom_idx) / tlG};
rowT = {tivE / tlE, cluster2elem(Gc, tivG, Ne, dom_idx) / tlG, cluster2elem(Gc, tivH, Ne, dom_idx) / tlG};
rowN = {'GN, element (whole domain)', sprintf('GN, %d clusters (%s)', Gc.N, MASK), sprintf('HK P1 (%s)', MASK)};
colN = {sprintf('start insp. (%.2f s)', t(k0)), sprintf('mid insp. (%.2f s)', t(k1)), ...
        sprintf('end insp. (%.2f s)', t(k2)), sprintf('TIV, mean of %d breaths', nb)};
f2 = newfig(W2, 5.0);
tl = tiledlayout(f2, 3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for r = 1:3
    Vr = {rowV{r}(k0), rowV{r}(k1), rowV{r}(k2), rowT{r}};
    av = cellfun(@(v) abs(v(isfinite(v))), Vr, 'UniformOutput', false);
    cl = prc(vertcat(av{:}), 99);  cl = [-cl cl];
    for c = 1:4
        ax = nexttile(tl);
        plot_slice(ax, SL, Vr{c}, cl, cmap, Rx);
        draw_edges(ax, SL.XY, SL.lung_edges, [0.25 0.25 0.25], 0.4);   % library lungs, for orientation
        if r == 1; title(ax, colN{c}, 'FontWeight', 'normal', 'FontSize', 7); end
        if c == 1; text(ax, -1.25*Rx, 0, rowN{r}, 'Rotation', 90, 'HorizontalAlignment', 'center', 'FontSize', 7); end
        if c == 4
            cb = colorbar(ax);  cb.Label.String = 'gas / mean TIV';  cb.FontSize = 7;
        end
    end
end
title(tl, sprintf('Neonate, breath %d of %d (median TV); prone, radiological view; colour limits = 99th pct per row', b0, nb), 'FontSize', 8);
save_fig(f2, fullfile(out, 'Fig02_images'), 'image');

% ======================= Fig 3: transport field =======================
[ei, ni, vv] = find(Gc.B);
plusN = zeros(Gc.E,1);  minusN = zeros(Gc.E,1);
plusN(ei(vv > 0)) = ni(vv > 0);  minusN(ei(vv < 0)) = ni(vv < 0);
into_ant = double(ant_c(minusN)) - double(ant_c(plusN));     % +1 if m > 0 crosses into anterior
Qchk = ((Gc.ell(:) .* into_ant)' * m)';
cq = corrcoef(Qchk, S2.QG(:));  sgn = sign(cq(1,2));
fprintf('flux sign check vs hk_pendelluft: corr %+.3f, rel diff %.2e (using sign %+d)\n', ...
        cq(1,2), norm(sgn*Qchk - S2.QG(:)) / norm(S2.QG(:)), sgn);
mid = xf((Gc.cent(plusN,:) + Gc.cent(minusN,:)) / 2);
dv  = xf(Gc.cent(minusN,:)) - xf(Gc.cent(plusN,:));  dv = dv ./ max(sqrt(sum(dv.^2, 2)), eps);
Mb = zeros(Gc.E, 3);  nM = zeros(1, 3);
for b = 1:nb
    st = kmin(b):kei(b)-1;  n = numel(st);  if n < 3; continue; end
    bin = min(floor(3 * ((0:n-1) + 0.5) / n) + 1, 3);
    for h = 1:3; Mb(:,h) = Mb(:,h) + sum(m(:, st(bin == h)), 2);  nM(h) = nM(h) + nnz(bin == h); end
end
Mb = sgn * (Gc.ell(:) .* Mb ./ max(nM, 1));                 % mean flux per edge (positive: plus -> minus)
% coarse-grain: all lung edges (whole 3-D lung) projected on the transverse
% plane, averaged in an ng x ng grid of the slice
ng = 11;  ed = linspace(-Rx, Rx, ng+1);  bw = ed(2) - ed(1);
ix = discretize(mid(:,1), ed);  iy = discretize(mid(:,2), ed);  okb = ~isnan(ix) & ~isnan(iy);
lin = sub2ind([ng ng], ix(okb), iy(okb));
cnt = accumarray(lin, 1, [ng*ng 1]);
cx  = accumarray(lin, mid(okb,1), [ng*ng 1]) ./ max(cnt, 1);
cy  = accumarray(lin, mid(okb,2), [ng*ng 1]) ./ max(cnt, 1);
UB = cell(1, 3);
for h = 1:3
    u = dv(okb,:) .* Mb(okb,h);
    UB{h} = [accumarray(lin, u(:,1), [ng*ng 1]), accumarray(lin, u(:,2), [ng*ng 1])] ./ max(cnt, 1);
end
good = cnt >= 3;
smax = max(cellfun(@(u) max(sqrt(sum(u(good,:).^2, 2))), UB));
scl  = 0.85 * bw / max(smax, eps);
f3 = newfig(W2, 2.6);
tl = tiledlayout(f3, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ttl = {'early inspiration', 'mid inspiration', 'late inspiration'};
for h = 1:3
    ax = nexttile(tl);  hold(ax, 'on');
    patch(ax, 'Faces', SL.tri, 'Vertices', SL.XY, 'FaceColor', [0.96 0.96 0.96], 'EdgeColor', 'none');
    patch(ax, 'Faces', SL.tri(ismember(SL.eidx, lung_idx), :), 'Vertices', SL.XY, 'FaceColor', [0.90 0.90 0.90], 'EdgeColor', 'none');
    draw_edges(ax, SL.XY, SL.bnd_edges, ink, 0.8);  draw_edges(ax, SL.XY, SL.lung_edges, gry, 0.6);
    plot(ax, [-1 1]*Rx, [0 0], '--', 'Color', ink, 'LineWidth', 0.7);
    u = UB{h} * scl;
    up = good & u(:,2) > 0;  dn = good & u(:,2) <= 0;
    quiver(ax, cx(up) - u(up,1)/2, cy(up) - u(up,2)/2, u(up,1), u(up,2), 0, 'Color', pal(2,:), 'LineWidth', 1.1, 'MaxHeadSize', 0.8);
    quiver(ax, cx(dn) - u(dn,1)/2, cy(dn) - u(dn,2)/2, u(dn,1), u(dn,2), 0, 'Color', pal(1,:), 'LineWidth', 1.1, 'MaxHeadSize', 0.8);
    axis(ax, 'equal', 'off');  xlim(ax, [-1.15 1.15]*Rx);  ylim(ax, [-1.15 1.15]*Rx);
    title(ax, ttl{h}, 'FontWeight', 'normal');
end
title(tl, sprintf(['HK transport (mean of %d breaths, 3-D domain edges projected, %dx%d grid, common scale); ' ...
                   'orange: towards anterior, blue: towards posterior'], nb, ng, ng), 'FontSize', 7);
save_fig(f3, fullfile(out, 'Fig03_flux'), 'image');

% ======================= Fig 4: time courses =======================
QG = S2.QG(:);  F = (diff(hV) / dt)';  a = (F' * QG) / (F' * F);  r = QG - a*F;
tm = t(1:end-1) + dt/2;
f4 = newfig(W2, 5.6);
tl = tiledlayout(f4, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl);  hold(ax, 'on');
plot(ax, t, 100*gV/TVm, 'Color', gry);  plot(ax, t, 100*hV/TVm, 'Color', pal(1,:));
plot(ax, t(kmin), 100*gV(kmin)/TVm, 'v', 'Color', ink, 'MarkerSize', 4, 'MarkerFaceColor', ink);
ylabel(ax, 'global gas (%TV)');  legend(ax, {'GN', 'HK', 'end-expiration'}, 'Location', 'eastoutside', 'Box', 'off');
shade_insp(ax, t, kmin, kei);
ax = nexttile(tl);  hold(ax, 'on');
plot(ax, t, 100*(Gc.vol(ant_c)'*rho(ant_c,:))/TVm, 'Color', pal(2,:));
plot(ax, t, 100*(Gc.vol(~ant_c)'*rho(~ant_c,:))/TVm, 'Color', pal(1,:));
ylabel(ax, 'HK regional (%TV)');  legend(ax, {'anterior (dependent)', 'posterior'}, 'Location', 'eastoutside', 'Box', 'off');
shade_insp(ax, t, kmin, kei);
ax3 = nexttile(tl);  ax = ax3;  hold(ax, 'on');
plot(ax, tm, 100*QG/TVm, 'Color', gry);  plot(ax, tm, 100*r/TVm, 'Color', pal(3,:));
yline(ax, 0, 'Color', ink, 'LineWidth', 0.5);
ylabel(ax, 'flux into ant. (%TV/s)');
legend(ax, {'raw Q_\Gamma', sprintf('quadrature r = Q_\\Gamma - aF (a = %+.4f)', a)}, 'Location', 'eastoutside', 'Box', 'off');
shade_insp(ax, t, kmin, kei);
ax = nexttile(tl);  hold(ax, 'on');
S2f = ldf('E2f_results');  S2e = ldf('E2e_results');
nmf = arrayfun(@(c) c.name, S2f.C, 'UniformOutput', false);  nme = arrayfun(@(c) c.name, S2e.C, 'UniformOutput', false);
pool = [cell2mat(S2f.Bq(strncmp(nmf, 'null', 4) | strncmp(nmf, 'mm', 2))); cell2mat(S2e.Bq(strncmp(nme, 'null', 4)))];
Bq_real = S2f.Bq{strcmp(nmf, 'real')};
bh = bar(ax, 1:nb, Bq_real, 'grouped', 'BarWidth', 0.8, 'EdgeColor', 'none');
bh(1).FaceColor = pal(2,:);  bh(2).FaceColor = pal(1,:);
p95 = [prc(pool(:,1), 95), prc(pool(:,2), 95)];
yline(ax, max(p95), ':', 'Color', ink, 'LineWidth', 1);
xlabel(ax, 'breath');  ylabel(ax, 'quadrature PV (%TV)');
legend(ax, {'post \rightarrow ant', 'ant \rightarrow post', sprintf('pooled null p95 (%.2f)', max(p95))}, 'Location', 'eastoutside', 'Box', 'off');
xlabel(ax3, 't (s)');
title(tl, 'Neonate: global and regional gas, interface flux, per-breath quadrature pendelluft volume', 'FontSize', 8);
save_fig(f4, fullfile(out, 'Fig04_timecourses'));

% ======================= Fig 5: dose-response on the real background =======================
S2b = ldf('E2b_results');
nmb = arrayfun(@(c) c.name, S2b.C, 'UniformOutput', false);  dsb = arrayfun(@(c) c.dose, S2b.C);
dse = arrayfun(@(c) c.dose, S2e.C);
incb = @(j) [0, arrayfun(@(d) median(S2b.Bp{find(dsb == d, 1)}(:,j) - S2b.Bp{1}(:,j), 'omitnan'), S2b.doses(:)')];
ince = @(j) [0, arrayfun(@(d) median(S2e.Bq{find(dse == d, 1)}(:,j) - S2e.Bq{1}(:,j), 'omitnan'), S2e.doses(:)')];
xb = [0 S2b.doses(:)'];  xe = [0 S2e.doses(:)'];
f5 = newfig(W2, 2.7);
tl = tiledlayout(f5, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
dirs = {'posterior \rightarrow anterior', 'anterior \rightarrow posterior'};
jHK = [1 2];  jRO = [6 5];                              % region-out of the LOSING region
for q = 1:2
    ax = nexttile(tl);  hold(ax, 'on');
    plot(ax, [0 20], [0 20], '--', 'Color', gry, 'LineWidth', 0.8);
    plot(ax, xb, incb(jHK(q)), '-o', 'Color', pal(1,:), 'MarkerFaceColor', pal(1,:), 'MarkerSize', 4);
    plot(ax, xe, ince(q),      '-s', 'Color', pal(3,:), 'MarkerFaceColor', pal(3,:), 'MarkerSize', 4);
    plot(ax, xb, incb(3),      '-^', 'Color', pal(4,:), 'MarkerFaceColor', pal(4,:), 'MarkerSize', 4);
    plot(ax, xb, incb(jRO(q)), '-d', 'Color', pal(5,:), 'MarkerFaceColor', pal(5,:), 'MarkerSize', 4);
    xlabel(ax, 'injected transfer (%TV per inspiration)');  ylabel(ax, 'estimated increment (%TV)');
    yline(ax, 0, 'Color', gry, 'LineWidth', 0.5, 'HandleVisibility', 'off');
    title(ax, dirs{q}, 'FontWeight', 'normal');  axis(ax, [0 20, min([0, incb(jHK(q)), ince(q), incb(3), incb(jRO(q))]) - 0.5, 20]);   % show negative increments  grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
    if q == 2
        legend(ax, {'identity', 'HK raw', 'HK quadrature', 'FRIC-vol (GN)', 'region-out (GN)'}, 'Location', 'northwest', 'Box', 'off');
    end
end
title(tl, sprintf('Sensitivity on the real neonate background (median over %d breaths)', nb), 'FontSize', 8);
save_fig(f5, fullfile(out, 'Fig05_dose_response'));

% ======================= Fig 6: false-positive floors =======================
medc = @(B) median(B, 1, 'omitnan');
pick = @(S, nm, pat, fld) max(cell2mat(cellfun(medc, S.(fld)(strncmp(nm, pat, numel(pat))), 'UniformOutput', false)), [], 1);
FL_raw = [pick(S2f, nmf, 'null-w', 'Braw'); pick(S2f, nmf, 'mm', 'Braw'); pick(S2e, nme, 'null-g', 'Braw'); medc(S2f.Braw{1})];
FL_q   = [pick(S2f, nmf, 'null-w', 'Bq');   pick(S2f, nmf, 'mm', 'Bq');   pick(S2e, nme, 'null-g', 'Bq');   medc(S2f.Bq{1})];
cats = {'noise', 'model mismatch \pm10%', 'w amplitude \pm30%', 'REAL'};
f6 = newfig(W2, 2.6);
tl = tiledlayout(f6, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for q = 1:2
    ax = nexttile(tl);
    bb = bar(ax, 1:4, max([FL_raw(:,q) FL_q(:,q)], 0.01), 'grouped', 'BaseValue', 0.01, 'EdgeColor', 'none');
    set(ax, 'XTick', 1:4, 'XTickLabel', cats);
    bb(1).FaceColor = [0.7 0.7 0.7];  bb(2).FaceColor = pal(3,:);
    set(ax, 'YScale', 'log');  ylim(ax, [0.01 20]);  ylabel(ax, 'PV (%TV, median of breaths)');
    title(ax, dirs{q}, 'FontWeight', 'normal');  grid(ax, 'on');  ax.YMinorGrid = 'off';  ax.GridColor = [0.85 0.85 0.85];
    for i = 1:4
        text(ax, i - 0.14, max(FL_raw(i,q), 0.01)*1.25, sprintf('%.2f', FL_raw(i,q)), 'HorizontalAlignment', 'center', 'FontSize', 6);
        text(ax, i + 0.14, max(FL_q(i,q), 0.01)*1.25,   sprintf('%.2f', FL_q(i,q)),   'HorizontalAlignment', 'center', 'FontSize', 6);
    end
    if q == 2; legend(ax, {'raw interface flux', 'quadrature'}, 'Location', 'northwest', 'Box', 'off'); end
end
title(tl, 'False-positive floors (zero-pendelluft nulls) vs the real neonate', 'FontSize', 8);
save_fig(f6, fullfile(out, 'Fig06_floors'));

% ======================= Fig 7: per-breath test =======================
f7 = newfig(W1, 3.0);
ax = axes(f7);  hold(ax, 'on');
rs = RandStream('mt19937ar', 'Seed', 7);
for q = 1:2
    x0 = 2*q - 1;
    jn = (rand(rs, size(pool,1), 1) - 0.5) * 0.35;  jr = (rand(rs, nb, 1) - 0.5) * 0.35;
    scatter(ax, x0 + jn, max(pool(:,q), 0.005), 14, gry, 'o');
    scatter(ax, x0 + 0.6 + jr, max(Bq_real(:,q), 0.005), 18, pal(2,:), 'filled');
    plot(ax, x0 + [-0.3 0.9], [1 1]*p95(q), ':', 'Color', ink);
end
set(ax, 'YScale', 'log', 'XTick', [1.3 3.3], 'XTickLabel', {'post\rightarrowant', 'ant\rightarrowpost'});
ylim(ax, [0.005 10]);  xlim(ax, [0.4 4.2]);  ylabel(ax, 'quadrature PV per breath (%TV)');
legend(ax, {sprintf('null breaths (n = %d)', size(pool,1)), sprintf('real breaths (n = %d)', nb), 'null p95'}, 'Location', 'southoutside', 'NumColumns', 2, 'Box', 'off');
title(ax, 'Real vs pooled null breaths', 'FontWeight', 'normal');
save_fig(f7, fullfile(out, 'Fig07_per_breath'));

% ======================= Fig 8: asynchrony calibration =======================
Sg = ldf('E2g_v2_results');  Sh = ldf('E2h_results');
tv1  = [-0.10; -0.05; 0.05; 0.10];                           % imposed delays of the v1 'prev' rows (lung mask only)
tauD = 1e3 * [Sg.Res(:,1); tv1(1:size(Sg.prev,1))];
pvD  = [Sg.Res(:,3:4); Sg.prev(:,2:3)];
tauD = [tauD; 0];  pvD = [pvD; medc(S2f.Bq{strcmp(nmf, 'null-w r1')})];   % zero asynchrony = in-phase null
[tauD, o] = sort(tauD);  pvD = pvD(o,:);
nmh  = {Sh.C.name};  isrc = ~strcmp(nmh, 'real');           % RC: + anterior slower, - posterior slower
tauR = cellfun(@(s) sscanf(s(strfind(s, 'tau')+3:end), '%f'), nmh(isrc))' .* (1 - 2*strncmp(nmh(isrc), 'post', 4))';
pvR  = Sh.R(isrc, 1:2);
[tauR, o] = sort(tauR);  pvR = pvR(o,:);
% stationary breaths (4-9) of the real data, quadrature, from E2e
qr = S2e.QGs{1}(:) - S2e.aa(1) * S2e.Fs{1}(:);
Bst = 100 * insp_pv(qr, kmin, kei, dt) ./ TVg;
real_all = medc(S2f.Bq{1});  real_st = medc(Bst(4:end, :));
f8 = newfig(W2, 2.7);
tl = tiledlayout(f8, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl);  hold(ax, 'on');
plot(ax, tauD, pvD(:,2), '-o', 'Color', pal(1,:), 'MarkerFaceColor', pal(1,:), 'MarkerSize', 4);
plot(ax, tauD, pvD(:,1), '-o', 'Color', pal(2,:), 'MarkerFaceColor', pal(2,:), 'MarkerSize', 4);
plot(ax, tauR, pvR(:,2), '--^', 'Color', pal(1,:), 'MarkerSize', 4);
plot(ax, tauR, pvR(:,1), '--^', 'Color', pal(2,:), 'MarkerSize', 4);
yline(ax, real_all(2), '-', 'real A\rightarrowP', 'Color', pal(1,:), 'LineWidth', 0.8, 'FontSize', 7, 'LabelHorizontalAlignment', 'left');
yline(ax, real_all(1), '-', 'real P\rightarrowA', 'Color', pal(2,:), 'LineWidth', 0.8, 'FontSize', 7, 'LabelHorizontalAlignment', 'left');
xline(ax, 0, 'Color', gry, 'LineWidth', 0.5);
xlabel(ax, 'imposed asynchrony (ms; > 0: anterior later / slower)');  ylabel(ax, 'quadrature PV in inspiration (%TV)');
legend(ax, {'delay, A\rightarrowP', 'delay, P\rightarrowA', 'RC, A\rightarrowP', 'RC, P\rightarrowA'}, 'Location', 'north', 'Box', 'off', 'NumColumns', 2);
title(ax, 'Transfer-free asynchrony nulls', 'FontWeight', 'normal');  grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
ax = nexttile(tl);  hold(ax, 'on');
nz = tauD ~= 0;
plot(ax, tauD(nz), pvD(nz,2) ./ pvD(nz,1), '-o', 'Color', ink, 'MarkerFaceColor', ink, 'MarkerSize', 4);
plot(ax, tauR, pvR(:,2) ./ pvR(:,1), '--^', 'Color', ink, 'MarkerSize', 4);
yline(ax, real_all(2)/real_all(1), '-', 'Color', pal(2,:));
yline(ax, real_st(2)/real_st(1), '--', 'Color', pal(2,:));
yline(ax, 1, ':', 'Color', gry);  xline(ax, 0, 'Color', gry, 'LineWidth', 0.5);
xlabel(ax, 'imposed asynchrony (ms)');  ylabel(ax, 'asymmetry A\rightarrowP / P\rightarrowA');
legend(ax, {'delay', 'RC time constant', sprintf('real, all breaths (%.2f)', real_all(2)/real_all(1)), ...
             sprintf('real, breaths 4-%d (%.2f)', nb, real_st(2)/real_st(1))}, 'Location', 'southeast', 'Box', 'off', 'FontSize', 7);
title(ax, 'Within-inspiration asymmetry', 'FontWeight', 'normal');  grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
save_fig(f8, fullfile(out, 'Fig08_asynchrony'));

% ======================= Fig 9: timing signature =======================
it = @(x) find(abs(Sg.taus - x) < 1e-9, 1);                 % rows of the measured-lag delay nulls
t1 = abs(Sg.tau_xc);  t2 = abs(Sg.tau_ph);
TTall = [Sg.TT(:)'; Sg.TTn(it(t1),:); Sg.TTn(it(t2),:); Sh.R(strcmp(nmh, 'ant tau 30 ms'),5:10); Sg.TTn(it(-t2),:)];
lab9 = {'real', sprintf('delay %+.1f ms', 1e3*t1), sprintf('delay %+.1f ms', 1e3*t2), 'RC ant. 30 ms', sprintf('delay %+.1f ms', -1e3*t2)};
xc = {'P\rightarrowA early', 'P\rightarrowA mid', 'P\rightarrowA late', 'A\rightarrowP early', 'A\rightarrowP mid', 'A\rightarrowP late'};
f9 = newfig(W2, 2.4);
ax = axes(f9);
bb = bar(ax, 1:6, TTall', 'grouped', 'EdgeColor', 'none');
set(ax, 'XTick', 1:6, 'XTickLabel', xc);
cols9 = [pal(2,:); pal(1,:); pal(3,:); pal(4,:); gry];
for i = 1:numel(bb); bb(i).FaceColor = cols9(i,:); end
ylabel(ax, 'quadrature PV (%TV)');  legend(ax, lab9, 'Location', 'northeastoutside', 'Box', 'off');
title(ax, 'Timing signature by thirds of inspiration (anterior later \Rightarrow A\rightarrowP early, P\rightarrowA late)', 'FontWeight', 'normal');
grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
save_fig(f9, fullfile(out, 'Fig09_timing'));

% ======================= Fig 10/11: adult simulation dose-response =======================
S4d = load(fullfile(here, 'E4d_results.mat'));  S4c = load(fullfile(here, 'E4c_results.mat'));
plot_E4(S4d.ALL, fieldnames(S4d.ALL), 'Adult thorax simulation, forward mesh finer than the inverse mesh (no inverse crime), 1% noise', ...
        fullfile(out, 'Fig10_E4d'), pal, gry, W2);
plot_E4(S4c.ALL, {'s1_n2', 's2_n2'}, 'Adult thorax simulation, same mesh (inverse crime), 1% noise', ...
        fullfile(out, 'Fig11_E4c'), pal, gry, W2);
% ======================= Fig 12: E5b piglets, local timing contrast =======================
f5b = dir(fullfile(here, 'E5b_results', '*.mat'));
if isempty(f5b)
    fprintf('Fig12 skipped: no E5b_results\n');
else
    R5 = [];
    for k = 1:numel(f5b); S = load(fullfile(f5b(k).folder, f5b(k).name)); R5 = [R5, S.R]; end %#ok<AGROW>
    SG  = [1 3 6 9 11];  LV = {'ZEEP start', 'PEEP10 inc', 'PEEP25', 'PEEP10 dec', 'ZEEP end'};   % by E5 segment: normal L1 is the END ZEEP
    nm3 = {'null1', 'null2', 'null3'};
    nC  = arrayfun(@(r) cellfun(@(n) r.(n).C, nm3), R5, 'UniformOutput', false);  nC = [nC{:}];
    nP  = arrayfun(@(r) cellfun(@(n) sum(r.(n).PV), nm3), R5, 'UniformOutput', false);  nP = [nP{:}];
    tC  = prc(abs(nC), 95);  tP = prc(nP, 95);
    ids5 = unique({R5.id});                                   % g08-c003, g08-c017, g10-c005, g10-c019
    f12 = newfig(W2, 3.3);
    tl = tiledlayout(f12, 1, 2, 'TileSpacing', 'compact', 'Padding', 'loose');   % compact clipped the x tick labels
    for pnl = 1:2
        ax = nexttile(tl);  hold(ax, 'on');
        hl = gobjects(numel(ids5), 1);  lab = cell(1, numel(ids5));
        for i = 1:numel(ids5)
            sel = R5(strcmp({R5.id}, ids5{i}));
            ist = find(strcmp(sel(1).state, {'normal', 'lavaged', 'surfactant'}));  lab{i} = [ids5{i} ' ' sel(1).state];
            col = pal(ist, :);  mk = 'o';  if strncmp(ids5{i}, 'g10', 3); mk = 's'; end
            x = arrayfun(@(r) find(SG == r.segment), sel) + (i - (numel(ids5) + 1)/2) * 0.1;
            if pnl == 1
                yr = arrayfun(@(r) r.real.C, sel);  yn = cell2mat(arrayfun(@(r) cellfun(@(n) r.(n).C, nm3)', sel, 'UniformOutput', false));
            else
                yr = arrayfun(@(r) sum(r.real.PV), sel);  yn = cell2mat(arrayfun(@(r) cellfun(@(n) sum(r.(n).PV), nm3)', sel, 'UniformOutput', false));
            end
            plot(ax, repmat(x, 3, 1), yn, '.', 'Color', gry, 'MarkerSize', 7, 'HandleVisibility', 'off');
            hl(i) = plot(ax, x, yr, mk, 'Color', col, 'MarkerFaceColor', col, 'MarkerSize', 6, 'LineStyle', 'none');
            % window untrustworthy if its own in-phase null exceeds the pooled |C| p95 -> open marker (excluded)
            bad = arrayfun(@(r) any(abs(cellfun(@(n) r.(n).C, nm3)) > tC), sel);
            plot(ax, x(bad), yr(bad), mk, 'Color', col, 'MarkerFaceColor', 'w', 'MarkerSize', 6, 'LineWidth', 1, 'HandleVisibility', 'off');
        end
        if pnl == 1
            yline(ax, 0, 'Color', gry, 'LineWidth', 0.5);
            yline(ax, [-tC tC], '--', 'Color', ink, 'LineWidth', 0.8);
            ylabel(ax, 'timing contrast C (%TV; > 0: dorsal later)');
            title(ax, sprintf('Timing contrast (null p95 |C| = %.2f)', tC), 'FontWeight', 'normal');
            lg = legend(ax, hl, lab, 'NumColumns', 3, 'Box', 'off');  lg.Layout.Tile = 'south';   % below both panels, never over data
        else
            yline(ax, tP, '--', 'Color', ink, 'LineWidth', 0.8);
            ylabel(ax, 'total quadrature PV (%TV)');
            title(ax, sprintf('Quadrature volume (null p95 = %.2f)', tP), 'FontWeight', 'normal');
        end
        set(ax, 'XTick', 1:5, 'XTickLabel', LV);  xlim(ax, [0.5 5.5]);  grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
    end
    title(tl, 'Piglets (E5b), PEEP-level-local HK: grey = 3 in-phase nulls, dashed = pooled null p95, open = excluded', 'FontSize', 8);
    save_fig(f12, fullfile(out, 'Fig12_E5b'));
end

% ======================= Fig 13: E3 paediatric ARDS, patient 1 =======================
% C from the saved median thirds TT (same definition as post_timing_shift); no
% circular-shift z: it is not calibrated for these windows (README)
f3e = dir(fullfile(here, 'E3_results', '1001_*_thorax_v2.mat'));
if isempty(f3e)
    fprintf('Fig13 skipped: no E3_results\\1001_*_thorax_v2.mat\n');
else
    E3 = cellfun(@(n) getfield(load(fullfile(here, 'E3_results', n)), 'res'), {f3e.name}, 'UniformOutput', false);   % cell: null_w only in some files
    Cof = @(TT) TT(1) + TT(6) - TT(4) - TT(3);
    PVt = @(B) sum(median(B, 1, 'omitnan'));
    rr  = cellfun(@(r) 60 * r.RR, E3);  pe  = cellfun(@(r) r.PEEP, E3);  dec = cellfun(@(r) strcmp(r.maneuver, 'decrement'), E3);
    hasn = cellfun(@(r) isfield(r, 'null_w'), E3);
    yv = {cellfun(@(r) Cof(r.real.TT), E3), cellfun(@(r) PVt(r.real.Bq), E3)};
    yn = {cellfun(@(r) Cof(r.null_w.TT), E3(hasn)), cellfun(@(r) PVt(r.null_w.Bq), E3(hasn))};
    f13 = newfig(W2, 3.1);
    tl = tiledlayout(f13, 1, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
    vaL = {'top', 'bottom'};                       % increment labels below, decrement above (both limbs meet at PEEP 14)
    for pnl = 1:2
        ax = nexttile(tl);  hold(ax, 'on');
        for lb = 0:1                               % 0: baseline + increment limb, 1: decrement limb
            s = find(dec == lb);  [~, o] = sort(pe(s), 'ascend');  if lb; o = flip(o); end;  s = s(o);
            plot(ax, pe(s), yv{pnl}(s), '-o', 'Color', pal(1 + lb, :), 'MarkerFaceColor', pal(1 + lb, :), 'MarkerSize', 5);
            text(ax, pe(s) + 0.4, yv{pnl}(s), compose('%.0f', rr(s)), 'FontSize', 6, 'Color', pal(1 + lb, :), ...
                 'VerticalAlignment', vaL{lb + 1});      % ventilator RR (/min): not constant across steps
        end
        plot(ax, pe(hasn), yn{pnl}, 'o', 'Color', gry, 'MarkerSize', 5);
        if pnl == 1
            tn = max(abs(yn{1}));
            yline(ax, 0, 'Color', gry, 'LineWidth', 0.5);  yline(ax, [-tn tn], '--', 'Color', ink, 'LineWidth', 0.8);
            ylabel(ax, 'timing contrast C (%TV; > 0: dorsal later)');
            title(ax, sprintf('Timing contrast (max |null C| = %.2f, %d nulls)', tn, nnz(hasn)), 'FontWeight', 'normal');
            lg = legend(ax, {'baseline + increment', 'decrement', 'in-phase null'}, 'NumColumns', 3, 'Box', 'off');  lg.Layout.Tile = 'south';
        else
            yline(ax, max(yn{2}), '--', 'Color', ink, 'LineWidth', 0.8);
            ylabel(ax, 'total quadrature PV (%TV)');
            title(ax, sprintf('Quadrature volume (max null = %.2f)', max(yn{2})), 'FontWeight', 'normal');
        end
        xlabel(ax, 'PEEP (cmH_2O)');  xlim(ax, [min(pe) - 1, max(pe) + 1]);  grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
    end
    title(tl, sprintf('Paediatric ARDS, patient 1 (E3, whole thorax, %d x 40 s, supine); labels = ventilator RR (/min)', numel(E3)), 'FontSize', 8);
    save_fig(f13, fullfile(out, 'Fig13_E3'));
end
% ======================= Fig 14: E6 mechanism / identifiability =======================
if ~exist(fullfile(here, 'E6_results.mat'), 'file')
    fprintf('Fig14 skipped: no E6_results.mat\n');
else
    S6 = load(fullfile(here, 'E6_results.mat'));
    M6 = squeeze(mean(S6.EST, 2));                       % conditions x [HKraw HKq GNopp qGN FRIC regout Sang lag]
    jE = [2 4 5 7 3];  nE = {'HK quadrature', 'GN quadrature', 'FRIC-vol', 'Sang', 'opposite flow'};
    mech = {S6.COND.mech};  dose = [S6.COND.dose];
    mkr = repmat({'o'}, 1, numel(mech));  mkr(strcmp(mech, 'delay')) = {'s'};  mkr(strcmp(mech, 'rc')) = {'^'};
    mkr(strcmp(mech, 'delay') & dose > 0) = {'d'};
    nM = {'transfer', 'delay', 'RC', 'delay + transfer'};  mM = {'o', 's', '^', 'd'};
    f14 = newfig(W2, 3.2);
    tl = tiledlayout(f14, 1, 2, 'TileSpacing', 'compact', 'Padding', 'loose');
    ax = nexttile(tl);  hold(ax, 'on');
    pos = S6.GT(:,2) > 0;                                 % M0 has GT 0: not on log axes (its values are in the caption)
    for j = 1:numel(jE)
        for ic = find(pos)'
            plot(ax, S6.GT(ic,2), max(M6(ic,jE(j)), 0.01), mkr{ic}, 'Color', pal(j,:), 'MarkerFaceColor', pal(j,:), ...
                 'MarkerSize', 5, 'HandleVisibility', 'off');
        end
    end
    plot(ax, [1 60], [1 60], '--', 'Color', ink, 'LineWidth', 0.8, 'HandleVisibility', 'off');
    for j = 1:numel(jE); plot(ax, NaN, NaN, 'o', 'Color', pal(j,:), 'MarkerFaceColor', pal(j,:), 'MarkerSize', 5); end
    for k = 1:numel(mM); plot(ax, NaN, NaN, mM{k}, 'Color', gry, 'MarkerFaceColor', gry, 'MarkerSize', 5); end
    set(ax, 'XScale', 'log', 'YScale', 'log', 'XTick', [2 5 10 20 50], 'YTick', [0.01 0.1 1 10]);  xlim(ax, [1.5 60]);  ylim(ax, [0.006 60]);
    xlabel(ax, 'true out-of-phase volume GT_q (%TV)');  ylabel(ax, 'estimate (%TV)');
    title(ax, 'Estimates vs ground truth (values < 0.01 drawn at 0.01)', 'FontWeight', 'normal');
    grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
    lg = legend(ax, [nE, nM], 'NumColumns', 5, 'Box', 'off');  lg.Layout.Tile = 'south';
    ax = nexttile(tl);  hold(ax, 'on');
    ia = ~strcmp(mech, 'none') | dose > 0;  ia(1) = false;
    for ic = find(ia)
        plot(ax, S6.GT(ic,3), M6(ic,8), mkr{ic}, 'Color', pal(2,:), 'MarkerFaceColor', pal(2,:), 'MarkerSize', 5);
    end
    lim = [-250 200];  plot(ax, lim, lim, '--', 'Color', ink, 'LineWidth', 0.8);
    xline(ax, 0, 'Color', gry, 'LineWidth', 0.5);  yline(ax, 0, 'Color', gry, 'LineWidth', 0.5);
    xlim(ax, lim);  ylim(ax, lim);  axis(ax, 'square');
    xlabel(ax, 'true regional lag, P behind A (ms)');  ylabel(ax, 'GN-estimated lag (ms)');
    title(ax, 'Lag (transfer: negative)', 'FontWeight', 'normal');
    grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
    title(tl, sprintf('E6 (adult model, no inverse crime, %d seeds): same estimators, three mechanisms', numel(S6.seeds)), 'FontSize', 8);
    save_fig(f14, fullfile(out, 'Fig14_E6'));
end

% ======================= Fig 15: E7 healthy adults, breath holds (in-vivo null) =======================
nm7 = {'draeger', 'swisstom', 'midas'};  f7 = cellfun(@(n) fullfile(here, ['E7_results_' n '.mat']), nm7, 'UniformOutput', false);
h7 = cellfun(@(p) exist(p, 'file') == 2, f7);
if ~h7(1)
    fprintf('Fig15 skipped: no E7_results_draeger.mat\n');
else
    S7 = cellfun(@load, f7(h7), 'UniformOutput', false);  n7 = nm7(h7);   % cell: the files differ in fields
    f15 = newfig(W2, 3.2);
    tl = tiledlayout(f15, 1, 3, 'TileSpacing', 'compact', 'Padding', 'loose');
    % (a) Draeger gas volume, holds shaded
    D7 = S7{1};  R = D7.E7.real;
    fps7 = 20;  if isfield(D7, 'fps'); fps7 = D7.fps; end      % shortcut: the Draeger v2 file predates the fps field (20 fps)
    tv = (0:numel(R.gV)-1) / fps7;  TVm = median(R.TV(R.valid));
    ax = nexttile(tl);  hold(ax, 'on');
    yl = [min(R.gV) max(R.gV)] / TVm + [-0.2 0.2];
    for h = 1:size(R.holds, 1)
        hp7 = patch(ax, R.holds(h, [1 2 2 1]) / fps7, yl([1 1 2 2]), [0.9 0.9 0.9], 'EdgeColor', 'none', 'HandleVisibility', 'off');
        uistack(hp7, 'bottom');
    end
    plot(ax, tv, R.gV / TVm, '-', 'Color', ink, 'LineWidth', 0.8);
    plot(ax, tv(R.kmin), R.gV(R.kmin) / TVm, 'o', 'Color', pal(1,:), 'MarkerSize', 3, 'MarkerFaceColor', pal(1,:));
    ylim(ax, yl);  xlim(ax, tv([1 end]));
    xlabel(ax, 'time (s)');  ylabel(ax, 'GN gas volume / median TV');
    title(ax, 'Draeger 20 fps: grey = breath holds', 'FontWeight', 'normal');
    % (b) HK quadrature: tidal breaths, hold windows, null breaths
    ax = nexttile(tl);  hold(ax, 'on');
    for i = 1:numel(S7)
        Rr = S7{i}.E7.real;  Rn = S7{i}.E7.null;
        yt = sum(Rr.Bq(Rr.valid,:), 2);  yn = sum(Rn.Bq(Rn.valid,:), 2);
        plot(ax, i - 0.22 + 0*yt, yt, 'o', 'Color', pal(1,:), 'MarkerFaceColor', pal(1,:), 'MarkerSize', 5);
        plot(ax, i + 0*Rr.holdHK, Rr.holdHK, 's', 'Color', pal(2,:), 'MarkerFaceColor', pal(2,:), 'MarkerSize', 5);
        plot(ax, i + 0.22 + 0*yn, yn, 'o', 'Color', gry, 'MarkerSize', 5);
    end
    set(ax, 'XTick', 1:numel(S7), 'XTickLabel', n7, 'YScale', 'log');  xlim(ax, [0.5 numel(S7) + 0.5]);  ylim(ax, [0.2 30]);
    ylabel(ax, 'HK quadrature volume (%TV)');  grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
    lg = legend(ax, {'tidal breaths', 'breath-hold windows', 'in-phase null breaths'}, 'NumColumns', 3, 'Box', 'off');
    lg.Layout.Tile = 'south';                              % inside the axes it covered the null points
    title(ax, 'Tidal vs breath hold (in-vivo null)', 'FontWeight', 'normal');
    % (c) Draeger: frames per breath
    ax = nexttile(tl);  hold(ax, 'on');
    rn = {'real', 'real10', 'real5', 'real2p5'};  rn = rn(isfield(D7.E7, rn));
    fpb = cellfun(@(r) median(diff(D7.E7.(r).kmin)), rn);
    yH = cellfun(@(r) sum(median(D7.E7.(r).Bq(D7.E7.(r).valid,:), 1, 'omitnan')), rn);
    yG = cellfun(@(r) median(D7.E7.(r).Bb(D7.E7.(r).valid, 2), 'omitnan'), rn);
    yB = cellfun(@(r) median(D7.E7.(r).holdHK), rn);
    plot(ax, fpb, yH, '-o', 'Color', pal(1,:), 'MarkerFaceColor', pal(1,:), 'MarkerSize', 5);
    plot(ax, fpb, yG, '-^', 'Color', pal(3,:), 'MarkerFaceColor', pal(3,:), 'MarkerSize', 5);
    plot(ax, fpb, yB, '-s', 'Color', pal(2,:), 'MarkerFaceColor', pal(2,:), 'MarkerSize', 5);
    xline(ax, 21, '--', 'Color', gry, 'HandleVisibility', 'off');
    text(ax, 22, 1.55*max([yH yG]), 'neonate E2', 'Color', gry, 'FontSize', 7, 'Rotation', 90, ...
         'HorizontalAlignment', 'right');                  % in the headroom above the curves, left of the legend
    set(ax, 'XScale', 'log');  ylim(ax, [0 1.6*max([yH yG])]);
    xlabel(ax, 'frames per breath');  ylabel(ax, 'median volume (%TV)');
    legend(ax, {'HK quad, tidal', 'GN quad, tidal', 'HK quad, holds'}, 'Location', 'northeast', 'Box', 'off', 'FontSize', 7);
    title(ax, 'Draeger, decimated', 'FontWeight', 'normal');  grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
    title(tl, 'E7: healthy adults, seated, whole thorax; breath holds = no flow', 'FontSize', 8);
    save_fig(f15, fullfile(out, 'Fig15_E7'));
end

% ======================= Fig 16: per-dataset gallery (real data) =======================
% one representative window per real dataset; maps from pend_maps (runs with MAPS_ONLY set)
GL = {'', 'Neonate, spontaneous';  'E3_1001_b', 'Paed. ARDS P1, baseline'; ...
      'E5b_g08-c003_S11', 'Piglet normal, end ZEEP';  'E5b_g08-c017_S11', 'Piglet lavaged, end ZEEP'; ...
      'E5b_g08-c035_S11', 'Piglet surfactant, end ZEEP';  'E7_draeger', 'Adult, Draeger (seated)';  'E7_midas', 'Adult, Midas (seated)'};
G16 = {};
for i = 1:size(GL, 1)
    if isempty(GL{i,1})                                    % neonate: from the E2 run already loaded
        G16(end+1,:) = {pend_maps(Gc, rgn, rho, m, kmin, kei, ant_c, dt), fm, Gc, GL{i,2}}; %#ok<AGROW>
    elseif exist(fullfile(here, 'maps', [GL{i,1} '.mat']), 'file')
        L16 = load(fullfile(here, 'maps', [GL{i,1} '.mat']));
        G16(end+1,:) = {L16.MP, L16.fm, L16.Gc, GL{i,2}}; %#ok<AGROW>
    end
end
nr = size(G16, 1);
seq = interp1([0 1], [1 1 1; 0.55 0.25 0.0], linspace(0, 1, 256));
% manual layout in inches (tiledlayout gave either wide gaps or, with uneven spans, a shrunken first column)
rh = 1.3;  yb = 0.35;  ms = 1.15;                         % row height, bottom margin (legend), map size
f16 = newfig(W2, yb + rh*nr + 0.6);
xm  = [0.30 1.50 3.20];  xcb = [NaN 2.70 4.40];           % map left edges, colourbar left edges
xc  = 5.10;  wc = W2 - xc - 0.08;                         % curves
cN = {'TIV, GN', 'TIV, HK', 'out-of-phase density, HK', 'regional gas (HK), / TV'};
for i = 1:nr
    [MPi, fmi, Gci, lbl] = G16{i,:};
    Nei = size(fmi.elems, 1);  domi = find(sum(Gci.EL, 2) > 0);
    [anti, rgti, cti] = axes_dir(fmi);
    Hi  = max(fmi.nodes(:,3)) - min(fmi.nodes(:,3));
    zi  = mean(cell2mat(arrayfun(@(e) fmi.nodes(e.nodes,3), fmi.electrode(:), 'UniformOutput', false)));
    xfi = @(P) [-(P(:,1:2) - cti) * rgti', (P(:,1:2) - cti) * anti'];
    SLi = slice_geom(fmi, zi + 1e-4*Hi, xfi);
    lung_i = [fmi.mat_idx{2}(:); fmi.mat_idx{3}(:)];
    SLi.lung_edges = boundary_edges(SLi.tri(ismember(SLi.eidx, lung_i), :));
    Rxi = max(abs(SLi.XY(:)));
    V3 = {cluster2elem(Gci, MPi.tivGN, Nei, domi), cluster2elem(Gci, MPi.tivHK, Nei, domi), cluster2elem(Gci, MPi.oopHK, Nei, domi)};
    clT = prc(abs([V3{1}(isfinite(V3{1})); V3{2}(isfinite(V3{2}))]), 99);
    y0 = yb + (nr - i) * rh;
    for c = 1:3
        ax = axes(f16, 'Units', 'inches', 'Position', [xm(c) y0+0.05 ms ms]);
        if c < 3; plot_slice(ax, SLi, V3{c}, [-clT clT], cmap, Rxi);
        else;     plot_slice(ax, SLi, V3{c}, [0 prc(V3{3}(isfinite(V3{3})), 99)], seq, Rxi); end
        draw_edges(ax, SLi.XY, SLi.lung_edges, [0.25 0.25 0.25], 0.4);
        if i == 1; title(ax, cN{c}, 'FontWeight', 'normal', 'FontSize', 7); end
        if c == 1; text(ax, -1.25*Rxi, 0, lbl, 'Rotation', 90, 'HorizontalAlignment', 'center', 'FontSize', 6.5); end
        if c >= 2
            cb = colorbar(ax);  cb.FontSize = 6;  cb.Units = 'inches';  cb.Position = [xcb(c) y0+0.12 0.08 ms-0.14];
            ax.Position = [xm(c) y0+0.05 ms ms];               % colorbar creation shrinks the axes: restore
        end
    end
    ax = axes(f16, 'Units', 'inches', 'Position', [xc y0+0.24 wc 0.82]);  hold(ax, 'on');
    yl = [min([MPi.VA - MPi.VA(MPi.kmin(1)), MPi.VP - MPi.VP(MPi.kmin(1))]), max([MPi.VA - MPi.VA(MPi.kmin(1)), MPi.VP - MPi.VP(MPi.kmin(1))])];
    for b = 1:numel(MPi.kei)
        hp16 = patch(ax, MPi.t([MPi.kmin(b) MPi.kei(b) MPi.kei(b) MPi.kmin(b)]), yl([1 1 2 2]), [0.93 0.93 0.93], 'EdgeColor', 'none', 'HandleVisibility', 'off');
        uistack(hp16, 'bottom');
    end
    plot(ax, MPi.t, MPi.VA - MPi.VA(MPi.kmin(1)), '-', 'Color', pal(1,:), 'LineWidth', 0.9);
    plot(ax, MPi.t, MPi.VP - MPi.VP(MPi.kmin(1)), '-', 'Color', pal(2,:), 'LineWidth', 0.9);
    xlim(ax, MPi.t([1 end]));  ylim(ax, yl);  ax.FontSize = 6;
    Bq16 = 100 * insp_pv(MPi.r * median(MPi.TVh), MPi.kmin, MPi.kei, MPi.dt) ./ MPi.TVg(:);   % per direction, as the run tables
    title(ax, sprintf('P>A %.1f / A>P %.1f %%TV (median, %d br.)', median(Bq16, 1, 'omitnan'), numel(MPi.kei)), 'FontWeight', 'normal', 'FontSize', 6.5);
    if i == 1
        lg = legend(ax, {'anterior region (HK)', 'posterior region (HK)'}, 'NumColumns', 2, 'Box', 'off', 'FontSize', 7);
        lg.ItemTokenSize = [14 8];  lg.Units = 'inches';  lg.Position = [W2/2-1.3 0.02 2.6 0.16];   % below all rows
    end
    if i == nr; xlabel(ax, 'time (s)'); end
end
sgtitle(f16, {'Per-dataset gallery: one window per real dataset (grey = inspirations; TIV and density / domain mean)', ...
           'curves: regional gas / median TV; titles: HK quadrature PV per direction, median over breaths, % of GN TV (as in the tables)'}, 'FontSize', 8);
save_fig(f16, fullfile(out, 'Fig16_gallery'), 'image');

fprintf('figures written to %s\n', out);

% ======================================================================
function f = newfig(w, h)
% white figure, light graphics theme (MATLAB R2025a+ otherwise follows the desktop dark theme)
f = figure('Units', 'inches', 'Position', [1 1 w h], 'Color', 'w');
try; theme(f, 'light'); catch; end
end

function plot_E4(ALL, keys, ttl, fname, pal, gry, W2)
f = newfig(W2, 2.5);
tl = tiledlayout(f, 1, numel(keys), 'TileSpacing', 'compact', 'Padding', 'compact');
mk = {'o', '^', 'v', 'd', 's'};  cc = [pal(1,:); pal(4,:); pal(4,:)*0.7; pal(5,:); pal(5,:)*0.7];
for q = 1:numel(keys)
    S = ALL.(keys{q});  ax = nexttile(tl);  hold(ax, 'on');
    x = 100 * mean(S.tab(:,:,1), 2);  xm = max(x);
    plot(ax, [0 xm], [0 xm], '--', 'Color', gry, 'LineWidth', 0.8);
    for j = 1:numel(S.meth)
        y = 100 * mean(S.tab(:,:,1+j), 2);
        plot(ax, x, y, ['-' mk{j}], 'Color', cc(j,:), 'MarkerFaceColor', cc(j,:), 'MarkerSize', 3.5);
    end
    axis(ax, [0 xm 0 1.25*xm]);  grid(ax, 'on');  ax.GridColor = [0.85 0.85 0.85];
    xlabel(ax, 'true PV (x 100, model units)');  if q == 1; ylabel(ax, 'estimated PV (x 100)'); end
    title(ax, S.scen, 'FontWeight', 'normal', 'FontSize', 7);
    if q == numel(keys); legend(ax, [{'identity'}, cellstr(S.meth(:))'], 'Location', 'northwest', 'Box', 'off', 'FontSize', 6); end
end
title(tl, ttl, 'FontSize', 8);
save_fig(f, fname);
end

function SL = slice_geom(fm, z, xf)
% 2-D slice of the 3-D thorax FEM; each triangle keeps the index of its parent element
img = mk_image(fm, (1:size(fm.elems,1))');
img.fwd_model.mdl_slice_mesher.interp_elems = false;
nimg = mdl_slice_mesher(img, [inf inf z]);
P = nimg.fwd_model.nodes(:, 1:2);  tri = nimg.fwd_model.elems;
[Pu, ~, ic] = uniquetol(P, 1e-9, 'ByRows', true, 'DataScale', 1);
SL.tri  = ic(tri);
SL.XY   = xf(Pu);
SL.eidx = round(nimg.elem_data(:));
SL.bnd_edges = boundary_edges(SL.tri);
end

function be = boundary_edges(tri)
e = sort([tri(:,[1 2]); tri(:,[2 3]); tri(:,[3 1])], 2);
[u, ~, j] = unique(e, 'rows');  cnt = accumarray(j, 1);
be = u(cnt == 1, :);
end

function draw_edges(ax, XY, E, col, lw)
X = [XY(E(:,1),1), XY(E(:,2),1), nan(size(E,1),1)]';  Y = [XY(E(:,1),2), XY(E(:,2),2), nan(size(E,1),1)]';
plot(ax, X(:), Y(:), '-', 'Color', col, 'LineWidth', lw);
end

function plot_slice(ax, SL, v, cl, cmap, Rx)
hold(ax, 'on');
patch(ax, 'Faces', SL.tri, 'Vertices', SL.XY, 'FaceColor', [0.90 0.90 0.90], 'EdgeColor', 'none');
cv = v(SL.eidx);  ok = isfinite(cv);
patch(ax, 'Faces', SL.tri(ok,:), 'Vertices', SL.XY, 'FaceVertexCData', cv(ok), 'FaceColor', 'flat', 'EdgeColor', 'none');
draw_edges(ax, SL.XY, SL.bnd_edges, [0.15 0.15 0.15], 0.7);
plot(ax, [-1 1]*Rx, [0 0], '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.5);
colormap(ax, cmap);  caxis(ax, cl);
axis(ax, 'equal', 'off');  xlim(ax, [-1.1 1.1]*Rx);  ylim(ax, [-1.1 1.1]*Rx);
end

function v = cluster2elem(Gc, x, Ne, lung_idx)
v = nan(Ne, 1);  ve = Gc.EL * x;  v(lung_idx) = ve(lung_idx);
end

function shade_insp(ax, t, kmin, kei)
yl = ylim(ax);
hp = gobjects(numel(kei), 1);
for b = 1:numel(kei)
    hp(b) = patch(ax, [t(kmin(b)) t(kei(b)) t(kei(b)) t(kmin(b))], [yl(1) yl(1) yl(2) yl(2)], [0.6 0.6 0.6], ...
                  'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end
ylim(ax, yl);  xlim(ax, [t(1) t(end)]);
uistack(hp, 'bottom');                                   % data on top of the shading
end

function cmap = diverging_map(n)
% blue (gas loss) - light neutral grey - red (gas gain)
c1 = [33 102 172]/255;  c0 = [0.96 0.96 0.96];  c2 = [178 24 43]/255;
h  = floor(n/2);  s = linspace(0, 1, h)';
cmap = [c1 + (c0 - c1).*s; c0 + (c2 - c0).*linspace(0, 1, n-h)'];
end

function save_fig(f, base, ctype)
% ctype 'vector' (line/bar plots) or 'image' (FEM patch maps: vector PDF would be huge)
if nargin < 3; ctype = 'vector'; end
drawnow;
savefig(f, [base '.fig']);
try
    exportgraphics(f, [base '.png'], 'Resolution', 300);
    if strcmp(ctype, 'image')
        exportgraphics(f, [base '.pdf'], 'ContentType', 'image', 'Resolution', 600);
    else
        exportgraphics(f, [base '.pdf'], 'ContentType', 'vector');
    end
catch
    print(f, [base '.png'], '-dpng', '-r300');  print(f, [base '.pdf'], '-dpdf', '-painters');
end
end

function B = insp_pv(q, kmin, kei, dt)
nb = numel(kei);  B = nan(nb, 2);
for b = 1:nb
    st = kmin(b):kei(b)-1;
    if numel(st) < 2; continue; end
    B(b,:) = [dt*sum(max(q(st), 0)), dt*sum(max(-q(st), 0))];
end
end

function q = prc(x, pc)
x = sort(x(isfinite(x)));  n = numel(x);  r = 1 + (n-1)*pc/100;  lo = floor(r);  hi = ceil(r);
q = x(lo) + (r - lo) * (x(hi) - x(lo));
end

function [ant, rgt, c_t] = axes_dir(fm)
ce  = interp_mesh(fm, 0);  vol = get_elem_volume(fm);
wc  = @(ix) sum(ce(ix,1:2) .* vol(ix), 1) / sum(vol(ix));
c_t = wc(1:size(fm.elems,1));  c_l = wc(fm.mat_idx{2});  c_r = wc(fm.mat_idx{3});
rgt = (c_r - c_l) / norm(c_r - c_l);
ant = c_t - (c_l + c_r)/2;  ant = ant - dot(ant, rgt)*rgt;  ant = ant / norm(ant);
end

function order = electrode_order(fm, towards)
[ant, rgt, c_t] = axes_dir(fm);
ep  = zeros(numel(fm.electrode), 2);
for k = 1:numel(fm.electrode); ep(k,:) = mean(fm.nodes(fm.electrode(k).nodes, 1:2), 1) - c_t; end
a   = atan2(ep*rgt', ep*ant');
[~, k0] = min(abs(a));  n = size(ep, 1);
order = circshift(1:n, -(k0-1));
side = dot(ep(order(5),:), rgt);
if (strcmp(towards, 'left') && side > 0) || (strcmp(towards, 'right') && side <= 0)
    order = order([1, n:-1:2]);
end
end

function [S, f] = spectrum(g, fps)
g = g(:)';  T = numel(g);
g = g - polyval(polyfit(1:T, g, 1), 1:T);
w = 0.5 - 0.5*cos(2*pi*(0:T-1)/(T-1));
nf = 8*2^nextpow2(T);
S = abs(fft(g .* w, nf));  f = (0:nf-1) / nf * fps;
keep = f <= fps/2;  S = S(keep);  f = f(keep);
end

function fpk = peak_in(S, f, band)
in = f >= band(1) & f <= band(2);  fi = f(in);  [~, k] = max(S(in));  fpk = fi(k);
end

function Y = fft_lowpass(X, fps, fc, tw)
T  = size(X, 2);
s  = (0:T-1) / (T-1);
tr = X(:,1) + (X(:,end) - X(:,1)) .* s;
f  = abs([0:floor(T/2), -ceil(T/2)+1:-1]) * fps / T;
H  = double(f <= fc - tw/2);
m  = f > fc - tw/2 & f < fc + tw/2;
H(m) = 0.5 * (1 + cos(pi * (f(m) - (fc - tw/2)) / tw));
Y  = real(ifft(fft(X - tr, [], 2) .* H, [], 2)) + tr;
end

function hp = lcurve_hp(A, Y)
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
kap([1:3, end-2:end]) = -Inf;
[~, k] = max(kap);  hp = sqrt(lams(k));
end

function X = gn_frames(A, Y, hp)
p   = 1 ./ sqrt(sum(A.^2, 1))';
PAt = p .* A';
X   = PAt * ((A*PAt + hp^2*eye(size(A,1))) \ Y);
end
