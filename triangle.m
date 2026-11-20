#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <simd/simd.h>
#include <math.h>

typedef struct {
    simd_float3 pos;
    simd_float3 col;
} Vertex;

typedef struct {
    simd_float4x4 mvp;
} Uniforms;

static const Vertex kVertices[24] = {
    {{-0.5f,-0.5f, 0.5f}, {1.0f,0.0f,0.0f}},
    {{ 0.5f,-0.5f, 0.5f}, {1.0f,0.0f,0.0f}},
    {{ 0.5f, 0.5f, 0.5f}, {1.0f,0.0f,0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {1.0f,0.0f,0.0f}},

    {{ 0.5f,-0.5f,-0.5f}, {0.0f,1.0f,0.0f}},
    {{-0.5f,-0.5f,-0.5f}, {0.0f,1.0f,0.0f}},
    {{-0.5f, 0.5f,-0.5f}, {0.0f,1.0f,0.0f}},
    {{ 0.5f, 0.5f,-0.5f}, {0.0f,1.0f,0.0f}},

    {{ 0.5f,-0.5f, 0.5f}, {0.0f,0.0f,1.0f}},
    {{ 0.5f,-0.5f,-0.5f}, {0.0f,0.0f,1.0f}},
    {{ 0.5f, 0.5f,-0.5f}, {0.0f,0.0f,1.0f}},
    {{ 0.5f, 0.5f, 0.5f}, {0.0f,0.0f,1.0f}},

    {{-0.5f,-0.5f,-0.5f}, {1.0f,1.0f,0.0f}},
    {{-0.5f,-0.5f, 0.5f}, {1.0f,1.0f,0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {1.0f,1.0f,0.0f}},
    {{-0.5f, 0.5f,-0.5f}, {1.0f,1.0f,0.0f}},

    {{-0.5f, 0.5f, 0.5f}, {0.0f,1.0f,1.0f}},
    {{ 0.5f, 0.5f, 0.5f}, {0.0f,1.0f,1.0f}},
    {{ 0.5f, 0.5f,-0.5f}, {0.0f,1.0f,1.0f}},
    {{-0.5f, 0.5f,-0.5f}, {0.0f,1.0f,1.0f}},

    {{-0.5f,-0.5f,-0.5f}, {1.0f,0.0f,1.0f}},
    {{ 0.5f,-0.5f,-0.5f}, {1.0f,0.0f,1.0f}},
    {{ 0.5f,-0.5f, 0.5f}, {1.0f,0.0f,1.0f}},
    {{-0.5f,-0.5f, 0.5f}, {1.0f,0.0f,1.0f}}
};

static const uint16_t kIndices[36] = {
     0, 1, 2,   0, 2, 3,
     4, 5, 6,   4, 6, 7,
     8, 9,10,   8,10,11,
    12,13,14,  12,14,15,
    16,17,18,  16,18,19,
    20,21,22,  20,22,23
};

static const char *kShaderSrc =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct V { float3 p; float3 c; };\n"
"struct U { float4x4 mvp; };\n"
"struct O { float4 pos [[position]]; float3 col; };\n"
"vertex O vs(const device V* verts [[buffer(0)]], constant U& u [[buffer(1)]], uint vid [[vertex_id]]) {\n"
"    O o;\n"
"    o.pos = u.mvp * float4(verts[vid].p, 1.0);\n"
"    o.col = verts[vid].c;\n"
"    return o;\n"
"}\n"
"fragment float4 fs(O in [[stage_in]]) {\n"
"    return float4(in.col, 1.0);\n"
"}\n";

static simd_float4x4 makePerspective(float fovy, float aspect, float nearZ, float farZ) {
    float ys = 1.0f / tanf(fovy * 0.5f);
    float xs = ys / aspect;
    float zs = farZ / (nearZ - farZ);
    return simd_matrix(
        simd_make_float4(xs, 0.0f, 0.0f, 0.0f),
        simd_make_float4(0.0f, ys, 0.0f, 0.0f),
        simd_make_float4(0.0f, 0.0f, zs, -1.0f),
        simd_make_float4(0.0f, 0.0f, zs * nearZ, 0.0f));
}

static simd_float4x4 makeLookAt(simd_float3 eye, simd_float3 center, simd_float3 up) {
    simd_float3 z = simd_normalize(eye - center);
    simd_float3 x = simd_normalize(simd_cross(up, z));
    simd_float3 y = simd_cross(z, x);
    return simd_matrix(
        simd_make_float4(x.x, y.x, z.x, 0.0f),
        simd_make_float4(x.y, y.y, z.y, 0.0f),
        simd_make_float4(x.z, y.z, z.z, 0.0f),
        simd_make_float4(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1.0f));
}

static simd_float4x4 makeRotationY(float r) {
    float c = cosf(r);
    float s = sinf(r);
    return simd_matrix(
        simd_make_float4(c, 0.0f, -s, 0.0f),
        simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f),
        simd_make_float4(s, 0.0f, c, 0.0f),
        simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f));
}

@interface Renderer : NSObject <MTKViewDelegate>
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLRenderPipelineState> pso;
@property (nonatomic, strong) id<MTLDepthStencilState> ds;
@property (nonatomic, strong) id<MTLBuffer> vb;
@property (nonatomic, strong) id<MTLBuffer> ib;
@property (nonatomic, strong) id<MTLBuffer> ub;
@property (nonatomic, assign) float elapsed;
@end

@implementation Renderer

- (instancetype)initWithView:(MTKView *)view {
    self = [super init];
    if (!self) {
        return nil;
    }

    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        NSLog(@"no metal device");
        return self;
    }

    view.device = _device;
    view.delegate = self;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    view.clearColor = MTLClearColorMake(0.08f, 0.08f, 0.12f, 1.0f);
    view.clearDepth = 1.0;

    _queue = [_device newCommandQueue];

    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:kShaderSrc];
    id<MTLLibrary> lib = [_device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        NSLog(@"shader error: %@", err);
        return self;
    }

    id<MTLFunction> vfn = [lib newFunctionWithName:@"vs"];
    id<MTLFunction> ffn = [lib newFunctionWithName:@"fs"];

    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vfn;
    pd.fragmentFunction = ffn;
    pd.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    pd.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
    _pso = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!_pso) {
        NSLog(@"pipeline error: %@", err);
        return self;
    }

    MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled = YES;
    _ds = [_device newDepthStencilStateWithDescriptor:dd];

    _vb = [_device newBufferWithBytes:kVertices
                               length:sizeof(kVertices)
                              options:MTLResourceStorageModeShared];
    _ib = [_device newBufferWithBytes:kIndices
                               length:sizeof(kIndices)
                              options:MTLResourceStorageModeShared];
    _ub = [_device newBufferWithLength:sizeof(Uniforms)
                               options:MTLResourceStorageModeShared];

    _elapsed = 0.0f;

    return self;
}

- (void)drawInMTKView:(MTKView *)view {
    if (!_pso || !_ds) {
        return;
    }
    if (!view.currentDrawable || !view.currentRenderPassDescriptor) {
        return;
    }

    _elapsed += 1.0f / 60.0f;

    CGSize sz = view.drawableSize;
    if (sz.height <= 0.0f) {
        return;
    }
    float aspect = (float)sz.width / (float)sz.height;

    simd_float4x4 proj = makePerspective(60.0f * (float)M_PI / 180.0f, aspect, 0.1f, 100.0f);
    simd_float4x4 vw = makeLookAt(simd_make_float3(2.5f, 2.0f, 2.5f),
                                  simd_make_float3(0.0f, 0.0f, 0.0f),
                                  simd_make_float3(0.0f, 1.0f, 0.0f));
    simd_float4x4 md = makeRotationY(_elapsed);

    Uniforms u;
    u.mvp = simd_mul(proj, simd_mul(vw, md));
    memcpy([_ub contents], &u, sizeof(u));

    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:_pso];
    [enc setDepthStencilState:_ds];
    [enc setCullMode:MTLCullModeBack];
    [enc setFrontFacingWinding:MTLWindingCounterClockwise];
    [enc setVertexBuffer:_vb offset:0 atIndex:0];
    [enc setVertexBuffer:_ub offset:0 atIndex:1];
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                    indexCount:36
                     indexType:MTLIndexTypeUInt16
                   indexBuffer:_ib
             indexBufferOffset:0];
    [enc endEncoding];

    [cb presentDrawable:view.currentDrawable];
    [cb commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
}

@end

@interface VC : UIViewController
@property (nonatomic, strong) Renderer *renderer;
@end

@implementation VC

- (void)viewDidLoad {
    [super viewDidLoad];

    MTKView *mv = [[MTKView alloc] initWithFrame:self.view.bounds];
    mv.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    mv.preferredFramesPerSecond = 60;
    [self.view addSubview:mv];

    self.renderer = [[Renderer alloc] initWithView:mv];
}

@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([VC class]));
    }
}
