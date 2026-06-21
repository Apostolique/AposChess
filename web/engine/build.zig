const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The parity test runner: checks the Zig engine against the JS oracle
    // (../engine-parity.json). Run from web/engine: `zig build parity`.
    const parity = b.addExecutable(.{
        .name = "apos-parity",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/parity.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(parity);

    const run_parity = b.addRunArtifact(parity);
    run_parity.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_parity.addArgs(args);
    const parity_step = b.step("parity", "Run the parity test against the JS oracle");
    parity_step.dependOn(&run_parity.step);

    // Search bench: `zig build bench -- --depth=8 [--nn]`.
    const bench = b.addExecutable(.{
        .name = "apos-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main_bench.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(bench);
    const run_bench = b.addRunArtifact(bench);
    run_bench.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_bench.addArgs(args);
    const bench_step = b.step("bench", "Search a position to a fixed depth and report nps");
    bench_step.dependOn(&run_bench.step);

    // Self-play match: `zig build match -- --games=20 --depth=6 [--nn]`.
    const match = b.addExecutable(.{
        .name = "apos-match",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main_match.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(match);
    const run_match = b.addRunArtifact(match);
    run_match.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_match.addArgs(args);
    const match_step = b.step("match", "Play self-play games and report results + nps");
    match_step.dependOn(&run_match.step);

    // Self-play data generator: `zig build gen -- --games=200 --depth=4 --eval=nn`.
    const gen = b.addExecutable(.{
        .name = "apos-gen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main_gen.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(gen);
    const run_gen = b.addRunArtifact(gen);
    run_gen.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_gen.addArgs(args);
    const gen_step = b.step("gen", "Generate self-play training data (JSONL)");
    gen_step.dependOn(&run_gen.step);

    // Browser target: the same core compiled to wasm32-freestanding, functions
    // exported over linear memory. `zig build wasm` -> zig-out/bin/apos.wasm.
    const wasm = b.addExecutable(.{
        .name = "apos",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wasm.zig"),
            .target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding }),
            .optimize = .ReleaseSmall,
        }),
    });
    wasm.entry = .disabled; // a library of exports, no main()
    wasm.rdynamic = true; // keep the exported functions
    const wasm_step = b.step("wasm", "Build the wasm32 module for the browser");
    wasm_step.dependOn(&b.addInstallArtifact(wasm, .{}).step);
}
