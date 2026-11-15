function D = load_real_eit(fname, opts)
%LOAD_REAL_EIT  Read one EIT recording into a standard struct + basic QC.
%
%   D = LOAD_REAL_EIT(fname)
%   D = LOAD_REAL_EIT(fname, opts)
%
%   opts (all optional)
%     .format  EIDORS format for eidors_readdata. Default from extension:
%              .get -> 'GET' (Goettingen/Viasys MCEIT), .eit -> 'EIT'
%              (auto-detect Draeger/Carefusion/Swisstom), .vtm -> 'VTM'.
%     .fps     frame rate to ASSUME if neither the reader nor a .prl file
%              provides one (default 13, Viasys default, also used by
%              Gomez-Laberge's EITReadData). D.fps_source records where the
%              value came from: 'reader' | 'prl' | 'assumed'.
%     .frames  frame indices to keep ([] = all)
%     .save    .mat path to save D to ('' = do not save)
%
%   GET/MCEIT data: eidors_readdata "untwists" the frames into a 256-row
%   (16 stim x 16 meas) layout in which the 48 rows touching a current
%   electrode are always zero, but returns a stim pattern describing only
%   the 208 measured rows. We keep exactly those 208 rows (meas_select of
%   the same 'no_meas_current','no_rotate_meas' pattern) and VERIFY that
%   every discarded row is zero, so D.vv matches D.stim row for row.
%
%   D fields
%     .file .format .vv (M x T) .stim .fps .fps_source .nframes .nmeas
%     .prl (parsed .prl key/values, [] if none) .aux (reader auxdata)
%     .qc  REPORT only - nothing is removed or modified:
%       .bad_frames   frames with NaN/Inf or all zeros
%       .chan_med     median |v| per measurement channel
%       .chan_flag    channels with median |v| < 1e-3x or > 1e3x the median
%                     over channels (possible electrode-contact problem)
%       .jump_frames  frames whose change from the previous frame exceeds
%                     10x the median frame-to-frame change (motion/artefact)
%       .prl_mismatch true if the .prl frame count differs from the file
%
%   Needs EIDORS on the path (eidors_readdata, mk_stim_patterns).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

if nargin < 2 || isempty(opts); opts = struct(); end
if ~isfield(opts, 'fps');    opts.fps = 13;   end
if ~isfield(opts, 'frames'); opts.frames = []; end
if ~isfield(opts, 'save');   opts.save = '';  end
if exist(fname, 'file') ~= 2
    error('load_real_eit: file not found: %s', fname);
end

[fdir, fbase, fext] = fileparts(fname);
if ~isfield(opts, 'format') || isempty(opts.format)
    switch lower(fext)
        case '.get';  opts.format = 'GET';
        case '.eit';  opts.format = 'EIT';
        case '.vtm';  opts.format = 'VTM';
        otherwise
            error('load_real_eit: unknown extension "%s" - pass opts.format', fext);
    end
end

% ---- read ------------------------------------------------------------
[vv, aux, stim] = eidors_readdata(fname, opts.format);

if any(strcmpi(opts.format, {'GET', 'MCEIT'}))
    [stim, msel] = mk_stim_patterns(16, 1, [0,1], [0,1], ...
                                    {'no_meas_current', 'no_rotate_meas'}, 1);
    if size(vv, 1) ~= numel(msel)
        error('load_real_eit: expected %d untwisted rows, got %d', numel(msel), size(vv,1));
    end
    if any(any(vv(~msel, :)))
        error('load_real_eit: rows outside meas_select are not zero - layout mismatch');
    end
    vv = vv(msel, :);                              % 256 -> 208, stim order
end
nm = sum(arrayfun(@(s) size(s.meas_pattern, 1), stim));
if size(vv, 1) ~= nm
    error('load_real_eit: %d data rows but the stim pattern has %d measurements', ...
          size(vv,1), nm);
end

if ~isempty(opts.frames)
    vv = vv(:, opts.frames);
end
vv = double(vv);
[M, T] = size(vv);

% ---- frame rate: reader -> .prl -> assumed ----------------------------
D.prl = [];
fps = NaN;  src = '';
if isstruct(aux) && isfield(aux, 'frame_rate') && isnumeric(aux.frame_rate) && ...
        isscalar(aux.frame_rate) && isfinite(aux.frame_rate)
    fps = aux.frame_rate;  src = 'reader';
end
prl_file = find_prl(fdir, fbase);
if ~isempty(prl_file)
    D.prl = parse_prl(prl_file);
    if ~isfinite(fps) && isfield(D.prl, 'framerate') && isnumeric(D.prl.framerate)
        fps = D.prl.framerate;  src = 'prl';
    end
end
if ~isfinite(fps)
    fps = opts.fps;  src = 'assumed';
    warning('load_real_eit:fps', ['%s: no frame rate in file or .prl - ASSUMING %g fps. ' ...
            'Verify from the data (e.g. cardiac peak vs known heart rate).'], ...
            [fbase fext], fps);
end

% ---- QC (report only) -------------------------------------------------
qc.bad_frames = find(any(~isfinite(vv), 1) | all(vv == 0, 1));
qc.chan_med   = median(abs(vv), 2);
cm            = median(qc.chan_med);
qc.chan_flag  = find(qc.chan_med < 1e-3*cm | qc.chan_med > 1e3*cm);
dv            = sqrt(sum(abs(diff(vv, 1, 2)).^2, 1));     % abs: complex data
qc.jump_frames = 1 + find(dv > 10*median(dv));
qc.prl_mismatch = ~isempty(D.prl) && isfield(D.prl, 'number_of_frames') && ...
                  isnumeric(D.prl.number_of_frames) && isempty(opts.frames) && ...
                  D.prl.number_of_frames ~= T;
if qc.prl_mismatch
    warning('load_real_eit:prl', ['%s: .prl says %d frames, file has %d. ' ...
            'The file may be only a segment (e.g. a reference recording).'], ...
            [fbase fext], D.prl.number_of_frames, T);
end

D.file = fname;   D.format = opts.format;
D.vv = vv;        D.stim = stim;
D.fps = fps;      D.fps_source = src;
D.nframes = T;    D.nmeas = M;
D.aux = aux;      D.qc = qc;

fprintf(['load_real_eit: %s | %d meas x %d frames | %.4g fps (%s) | ' ...
         'bad frames %d, flagged channels %d, jumps %d\n'], [fbase fext], M, T, ...
        fps, src, numel(qc.bad_frames), numel(qc.chan_flag), numel(qc.jump_frames));

if ~isempty(opts.save)
    save(opts.save, 'D');
end
end

% ======================================================================
function f = find_prl(fdir, fbase)
% exact <base>.prl (any case); else the longest .prl whose name is a prefix
% of the base name FOLLOWED BY '-' or '_' (e.g. CHB_CONV05.prl for
% CHB_CONV05-ref.get, but not X_1.prl for X_10.get)
f = '';
L = dir(fdir);
best = 0;
for k = 1:numel(L)
    if L(k).isdir; continue; end
    [~, nb, ex] = fileparts(L(k).name);
    if ~strcmpi(ex, '.prl'); continue; end
    if strcmpi(nb, fbase)
        f = fullfile(fdir, L(k).name);  return
    end
    n = numel(nb);
    if numel(fbase) > n && strncmpi(fbase, nb, n) && any(fbase(n+1) == '-_') && n > best
        best = n;  f = fullfile(fdir, L(k).name);
    end
end
end

function P = parse_prl(f)
% "Key words = value [unit words]" lines -> struct with lower_snake_case
% keys. A value is numeric only if it is a number optionally followed by
% unit WORDS ("13 per second", "5000000 Hz"); "02/11/2005" stays text.
P = struct();
txt = fileread(f);
lines = regexp(txt, '\r?\n', 'split');
for k = 1:numel(lines)
    tok = regexp(lines{k}, '^\s*([^=]+?)\s*=\s*(.*?)\s*$', 'tokens', 'once');
    if isempty(tok); continue; end
    key = lower(regexprep(strtrim(tok{1}), '[^A-Za-z0-9]+', '_'));
    key = regexprep(key, '^_+|_+$', '');
    if isempty(key) || ~isvarname(key); continue; end
    num = regexp(tok{2}, '^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?:\s+[A-Za-z].*)?$', ...
                 'tokens', 'once');
    if ~isempty(num); P.(key) = str2double(num{1}); else; P.(key) = tok{2}; end
end
end
