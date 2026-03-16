if ~exist("output_dir", "var")
    output_dir = "";
end

A = [1 1 1];
E = [64 64 64];
L = [4 4 6];
alpha = [acos(3 / 4), -acos(3 / 4), 0];

Dvec = [
    1 2 5 6
    5 6 3 4
    1 2 3 4
];

N = max(Dvec, [], "all");
I = eye(N);
S1 = I(Dvec(1, :), :);
S2 = I(Dvec(2, :), :);
S3 = I(Dvec(3, :), :);

kref = [
    1 0 -1 0
    0 0 0 0
    -1 0 1 0
    0 0 0 0
];

c = cos(alpha);
s = sin(alpha);

T1 = [
    c(1) s(1) 0 0
    -s(1) c(1) 0 0
    0 0 c(1) s(1)
    0 0 -s(1) c(1)
];
T2 = [
    c(2) s(2) 0 0
    -s(2) c(2) 0 0
    0 0 c(2) s(2)
    0 0 -s(2) c(2)
];
T3 = [
    c(3) s(3) 0 0
    -s(3) c(3) 0 0
    0 0 c(3) s(3)
    0 0 -s(3) c(3)
];

k1 = A(1) * E(1) / L(1) * kref;
k2 = A(2) * E(2) / L(2) * kref;
k3 = A(3) * E(3) / L(3) * kref;

kg1 = T1' * k1 * T1;
kg2 = T2' * k2 * T2;
kg3 = T3' * k3 * T3;

K = S1' * kg1 * S1 + S2' * kg2 * S2 + S3' * kg3 * S3;

G = [
    1 0 0 0 0 0
    0 1 0 0 0 0
    0 0 -sind(60) cosd(60) 0 0
];
Q = zeros(size(G, 1), 1);
a_G = max(K, [], "all");
Gbar = a_G * G;
Qbar = a_G * Q;

P = zeros(N, 1);
P(6) = -10;

AL = [K, Gbar'; Gbar, zeros(size(G, 1), size(G, 1))];
solution = AL \ [P; Qbar];

Lag = solution((N + 1):end) * a_G;
U = solution(1:N);
R = K(1:4, :) * U;

ueg1 = U(Dvec(1, :));
ueg2 = U(Dvec(2, :));
ueg3 = U(Dvec(3, :));

ue1 = T1 * ueg1;
ue2 = T2 * ueg2;
ue3 = T3 * ueg3;

local_displacements = [ue1, ue2, ue3];
member_forces = [k1 * ue1, k2 * ue2, k3 * ue3];
constraint_residual = G * U - Q;

if strlength(output_dir) > 0
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    writematrix(U, fullfile(output_dir, "U.tsv"), "Delimiter", "tab", "FileType", "text");
    writematrix(Lag, fullfile(output_dir, "Lag.tsv"), "Delimiter", "tab", "FileType", "text");
    writematrix(R, fullfile(output_dir, "R.tsv"), "Delimiter", "tab", "FileType", "text");
    writematrix(member_forces, fullfile(output_dir, "member_forces.tsv"), "Delimiter", "tab", "FileType", "text");
    writematrix(local_displacements, fullfile(output_dir, "local_displacements.tsv"), "Delimiter", "tab", "FileType", "text");
    writematrix(constraint_residual, fullfile(output_dir, "constraint_residual.tsv"), "Delimiter", "tab", "FileType", "text");
else
    result = struct( ...
        "U", U, ...
        "Lag", Lag, ...
        "R", R, ...
        "local_displacements", local_displacements, ...
        "member_forces", member_forces, ...
        "constraint_residual", constraint_residual ...
    );
    disp(result);
end
