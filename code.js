// 0 -> 3 sides
// 1 -> 6 sides
// 2 -> 12 sides
// 3 -> 24 sides
// 4 -> 48 sides
// 5 -> 96 sides
// etc.
let circleResolution = 3;

let numParticles = Math.round(2680); // must not be higher than 4194240 (i.e. 65535 * 64);
let maxParticleRadius = 0.02;
let minParticleRadius = maxParticleRadius;
let potentialCutoff = 2.5;
let simSize = 1.4;
let initialMaximumTimeStep = 0.001;
let initialInverseTimestep = 65536;
let timeStepCaution = 100;
let gravity = 1;
let initialTemperature = 1;

// determined by shader code
let bytesPerParticle = 40; // 8 byte aligned, apparently
let miscBufferLength = 20;

let gridCellsPerDimension = Math.floor(1 / (maxParticleRadius * potentialCutoff));
let gridCellSize = 1 / gridCellsPerDimension;
let gridCellCapacity = Math.ceil(2 * (gridCellSize / minParticleRadius + 1) ** 2);
let numGridCells = gridCellsPerDimension ** 2;
let gridBufferSize = numParticles + numGridCells * gridCellCapacity + 1;

let errorHasOccured = false;

let semiRandomState = 0;
function semiRandom() {
	let phi = (1 + Math.sqrt(5)) / 2;
	semiRandomState = (semiRandomState + phi) % 1;
	return semiRandomState;
}

function reportError(error) {
	errorHasOccured = true; // stop the main loop

	console.log("Error message:\n" + error.message + "\n\nStack traceback:\n" + error.stack);
	if (device) device.destroy();
}

main().catch(error => reportError(error));

async function main() {
	let shaderCode = await fetch("code.wgsl").then(x => x.text());

	shaderCode = shaderCode.replace("[[numParticles]]", numParticles);
	shaderCode = shaderCode.replace("[[maxParticleRadius]]", maxParticleRadius);
	shaderCode = shaderCode.replace("[[minParticleRadius]]", minParticleRadius);
	shaderCode = shaderCode.replace("[[potentialCutoff]]", potentialCutoff);
	shaderCode = shaderCode.replace("[[timeStepCaution]]", timeStepCaution);
	shaderCode = shaderCode.replace("[[gravity]]", gravity);
	shaderCode = shaderCode.replace("[[gridCellsPerDimension]]", gridCellsPerDimension);
	shaderCode = shaderCode.replace("[[gridCellSize]]", gridCellSize);
	shaderCode = shaderCode.replace("[[gridCellCapacity]]", gridCellCapacity);
	shaderCode = shaderCode.replace("[[numGridCells]]", numGridCells);

	let canvas = document.getElementById("can");
	canvas.width = 512;
	canvas.height = 512;

	let adapter = await navigator.gpu?.requestAdapter({powerPreference: "high-performance"});
	device = await adapter?.requestDevice();

	if (!device) {
		throw new Error("WebGPU is not supported by this device/operating system/browser combination. Try using Google Chrome on Windows, macOS, or ChromeOS. You may also try hitting \"run\" again in case this error is spurious.");
	}

	device.addEventListener("uncapturederror", event => reportError(event.error));

	let context = canvas.getContext("webgpu");

	context.configure({
		device,
		format: navigator.gpu.getPreferredCanvasFormat()
	});

	let computeModule = device.createShaderModule({
		label: "compute module",
		code: shaderCode
	});

	let renderModule = device.createShaderModule({
		label: "render module",
		code: shaderCode
	});

	let computePipeline = device.createComputePipeline({
		label: "compute pipeline",
		layout: "auto",
		compute: {
			module: computeModule,
			entryPoint: "computeShader"
		}
	});

	let renderPipeline = device.createRenderPipeline({
		label: "render pipeline",
		layout: "auto",
		vertex: {
			module: renderModule,
			entryPoint: "vertexShader"
		},
		fragment: {
			module: renderModule,
			entryPoint: "fragmentShader",
			targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
		}
	});

	let floatsPerParticle = bytesPerParticle / 4;

	let workBuffer = new Float32Array(numParticles * bytesPerParticle);

	for (let i = 0; i < numParticles; i++) {
		let sqrt1 = Math.round(Math.sqrt(numParticles / Math.sqrt(0.75)));
		let sqrt2 = Math.round(Math.sqrt(numParticles * Math.sqrt(0.75)));
		let x = ((i % sqrt1) / sqrt1 - 0.5) * (1 - 2 * maxParticleRadius) * 2 + 1 * maxParticleRadius;
		let y = (Math.floor(i / sqrt1) / sqrt2 - 0.5) * (1 - 2 * maxParticleRadius) * 2 - 0.5 * maxParticleRadius;
		let vx = Math.sqrt(initialTemperature) * (Math.random() * 2 - 1);
		let vy = Math.sqrt(initialTemperature) * (Math.random() * 2 - 1);
		let radius = maxParticleRadius;

		if (i % sqrt1 % 2 == 1) {
			y += 1 / sqrt2 * (1 - 2 * maxParticleRadius);
		}

		workBuffer[floatsPerParticle * i + 0] = x / simSize;
		workBuffer[floatsPerParticle * i + 1] = y / simSize;
		workBuffer[floatsPerParticle * i + 2] = vx * simSize;
		workBuffer[floatsPerParticle * i + 3] = vy * simSize;
		// 4: acceleration.x: f32
		// 5: acceleration.y: f32
		// 6: potential: f32
		workBuffer[floatsPerParticle * i + 7] = radius / simSize;
		// 8: coordinationNumber: f32
	}

	let grid = new Float32Array(gridBufferSize * bytesPerParticle);
	let gridCounters = new Uint32Array(numGridCells + 1);

	for (let i = 0; i < numParticles; i++) {
		let particle = {};
		particle.x = (workBuffer[floatsPerParticle * i + 0] + 1) / 2;
		particle.y = (workBuffer[floatsPerParticle * i + 1] + 1) / 2;

		let gridX = Math.floor(particle.x / gridCellSize);
		let gridY = Math.floor(particle.y / gridCellSize);
		let gridCellNumber = gridX + gridCellsPerDimension * gridY;

		if (gridCellNumber >= 0 && gridCellNumber < numGridCells) {
			let gridCellOffset = gridCellNumber * gridCellCapacity;
			let numParticlesInCell = gridCounters[gridCellNumber]++;

			if (numParticlesInCell < gridCellCapacity) {
				let gridIndex = gridCellOffset + numParticlesInCell;

				for (let j = 0; j < floatsPerParticle; j++) {
					grid[floatsPerParticle * gridIndex + j] = workBuffer[floatsPerParticle * i + j];
				}
			} else {
				throw new Error("Attempt to place a particle in a full grid cell");
			}
		} else {
			throw new Error("Attempt to place a particle outside the simulation field");
		}
	}

	let misc = new ArrayBuffer(miscBufferLength);
	let miscF32 = new Float32Array(misc);
	let miscU32 = new Uint32Array(misc);

	miscF32[0] = initialMaximumTimeStep;
	miscU32[1] = initialInverseTimestep;
	miscF32[2] = initialMaximumTimeStep;

	let workBuffer1 = device.createBuffer({
		label: "work buffer 1",
		size: workBuffer.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	let workBuffer2 = device.createBuffer({
		label: "work buffer 2",
		size: workBuffer.byteLength,
		usage: GPUBufferUsage.STORAGE
	});

	let gridBuffer1 = device.createBuffer({
		label: "grid buffer 1",
		size: grid.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	let gridBuffer2 = device.createBuffer({
		label: "grid buffer 2",
		size: grid.byteLength,
		usage: GPUBufferUsage.STORAGE
	});

	let gridCountersBuffer1 = device.createBuffer({
		label: "grid counters buffer 1",
		size: gridCounters.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	let gridCountersBuffer2 = device.createBuffer({
		label: "grid counters buffer 2",
		size: gridCounters.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	let miscBuffer1 = device.createBuffer({
		label: "misc buffer 1",
		size: misc.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	let miscBuffer2 = device.createBuffer({
		label: "misc buffer 2",
		size: misc.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	device.queue.writeBuffer(workBuffer1, 0, workBuffer);
	device.queue.writeBuffer(gridBuffer1, 0, grid);
	device.queue.writeBuffer(miscBuffer1, 0, misc);

	let computeBindGroup1 = device.createBindGroup({
		label: "compute bind group 1",
		layout: computePipeline.getBindGroupLayout(0),
		entries: [{
			binding: 0,
			resource: {
				buffer: workBuffer1
			}
		}, {
			binding: 1,
			resource: {
				buffer: gridBuffer1
			}
		}, {
			binding: 2,
			resource: {
				buffer: gridCountersBuffer1
			}
		}, {
			binding: 3,
			resource: {
				buffer: miscBuffer1
			}
		}, {
			binding: 4,
			resource: {
				buffer: workBuffer2
			}
		}, {
			binding: 5,
			resource: {
				buffer: gridBuffer2
			}
		}, {
			binding: 6,
			resource: {
				buffer: gridCountersBuffer2
			}
		}, {
			binding: 7,
			resource: {
				buffer: miscBuffer2
			}
		}]
	});

	// some buffers are swapped
	let computeBindGroup2 = device.createBindGroup({
		label: "compute bind group 2",
		layout: computePipeline.getBindGroupLayout(0),
		entries: [{
			binding: 0,
			resource: {
				buffer: workBuffer2
			}
		}, {
			binding: 1,
			resource: {
				buffer: gridBuffer2
			}
		}, {
			binding: 2,
			resource: {
				buffer: gridCountersBuffer2
			}
		}, {
			binding: 3,
			resource: {
				buffer: miscBuffer2
			}
		}, {
			binding: 4,
			resource: {
				buffer: workBuffer1
			}
		}, {
			binding: 5,
			resource: {
				buffer: gridBuffer1
			}
		}, {
			binding: 6,
			resource: {
				buffer: gridCountersBuffer1
			}
		}, {
			binding: 7,
			resource: {
				buffer: miscBuffer1
			}
		}]
	});

	let renderBindGroup = device.createBindGroup({
		label: "render bind group",
		layout: renderPipeline.getBindGroupLayout(0),
		entries: [{
			binding: 0,
			resource: {
				buffer: workBuffer1
			}
		}]
	});

	let renderPassDescriptor = {
		label: "render pass descriptor",
		colorAttachments: [{
			view: undefined, // assigned at render time
			clearValue: [0.0, 0.0, 0.0, 1.0],
			loadOp: "clear",
			storeOp: "store"
		}]
	};
let x = 0;
	function compute(iterations) {
		if (errorHasOccured) return;

		let encoder = device.createCommandEncoder({label: "compute encoder"});

		let pass = null;

		for (let i = 0; i < iterations; i++) {
			encoder.clearBuffer(gridCountersBuffer2);
			encoder.clearBuffer(miscBuffer2);

			pass = encoder.beginComputePass({label: "compute pass 1"});
			pass.setPipeline(computePipeline);
			pass.setBindGroup(0, computeBindGroup1);
			pass.dispatchWorkgroups(Math.ceil(numParticles / 64));
			pass.end();

			encoder.clearBuffer(gridCountersBuffer1);
			encoder.clearBuffer(miscBuffer1);

			pass = encoder.beginComputePass({label: "compute pass 2"});
			pass.setPipeline(computePipeline);
			pass.setBindGroup(0, computeBindGroup2);
			pass.dispatchWorkgroups(Math.ceil(numParticles / 64));
			pass.end();
		}

		let commandBuffer = encoder.finish();
		device.queue.submit([commandBuffer]);
	}


	function render() {
		if (errorHasOccured) return;

		encoder = device.createCommandEncoder({
			label: "render encoder"
		});

		renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();

		pass = encoder.beginRenderPass(renderPassDescriptor);

		pass.setPipeline(renderPipeline);
		pass.setBindGroup(0, renderBindGroup);
		pass.draw(Math.pow(2, circleResolution) * 9 - 6, numParticles); // the vertex shader takes care of constructing the circle when given ever higher vertex indices
		pass.end();

		commandBuffer = encoder.finish();
		device.queue.submit([commandBuffer]);
	}

	let observer = new ResizeObserver(entries => {
		for (let entry of entries) {
			let canvas = entry.target;
			let contentBoxSize = entry.contentBoxSize[0];
			let width = contentBoxSize.inlineSize;
			let height = contentBoxSize.blockSize;
			// 2x multiplier for antialiasing
			canvas.width = canvas.height = 2 * Math.min(width, height, device.limits.maxTextureDimension2D);

			render();
		}
	});

	observer.observe(canvas);

	async function mainLoop() {
		// increase this number if you have a beefy GPU
		compute(100);
		render();

		if (!errorHasOccured) {
			device.queue.onSubmittedWorkDone().then(mainLoop);
		}
	}

	mainLoop();
}

