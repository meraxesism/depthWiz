import { DEFAULT_GRID_SIZE } from './pipeline'

const modelId = 'onnx-community/depth-anything-v2-small'

type DepthImage = { data: ArrayLike<number>; width?: number; height?: number }
type DepthEstimator = (image: DepthImage) => Promise<{ depth?: DepthImage; predicted_depth?: { data: ArrayLike<number> } }>
let estimator: DepthEstimator | null = null

export const depthAnythingModel = modelId

export const estimateRelativeDepth = async (file: Blob, width = DEFAULT_GRID_SIZE, height = DEFAULT_GRID_SIZE): Promise<number[]> => {
  if (!estimator) {
    const { env, pipeline } = await import('@huggingface/transformers')
    env.allowLocalModels = false
    env.useBrowserCache = true
    try {
      estimator = await pipeline('depth-estimation', modelId, { device: 'webgpu' }) as unknown as DepthEstimator
    } catch {
      estimator = await pipeline('depth-estimation', modelId, { device: 'wasm' }) as unknown as DepthEstimator
    }
  }
  const { RawImage } = await import('@huggingface/transformers')
  const image = await RawImage.fromBlob(file)
  const result = await estimator(image)
  const depthData = result.depth?.data ?? result.predicted_depth?.data
  if (!depthData) throw new Error('Depth Anything V2 returned no depth map')
  const source = Array.from(depthData, Number)
  const sourceWidth = result.depth?.width ?? Math.round(Math.sqrt(source.length))
  const sourceHeight = result.depth?.height ?? sourceWidth
  const minimum = Math.min(...source)
  const maximum = Math.max(...source)
  const range = Math.max(1e-6, maximum - minimum)
  const normalized = source.map((value) => (value - minimum) / range)
  const resized = new Array<number>(width * height)
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const sourceRow = Math.min(sourceHeight - 1, Math.floor((row / height) * sourceHeight))
      const sourceColumn = Math.min(sourceWidth - 1, Math.floor((column / width) * sourceWidth))
      resized[row * width + column] = normalized[sourceRow * sourceWidth + sourceColumn] ?? 0
    }
  }
  return resized
}
