#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>
#include <math.h>

typedef struct { simd_float3 position; simd_float3 color; } VPos;
typedef struct { simd_float4x4 mvp; } Uniforms;

static const VPos kVertices[24] = {
    {{-0.5f,-0.5f, 0.5f}, {1,0,0}}, {{ 0.5f,-0.5f, 0.5f}, {1,0,0}},
    {{ 0.5f, 0.5f, 0.5f}, {1,0,0}}, {{-0.5f, 0.5f, 0.5f}, {1,0,0}},
    {{ 0.5f,-0.5f,-0.5f}, {0,1,0}}, {{-0.5f,-0.5f,-0.5f}, {0,1,0}},
    {{-0.5f, 0.5f,-0.5f}, {0,1,0}}, {{ 0.5f, 0.5f,-0.5f}, {0,1,0}},
    {{ 0.5f,-0.5f, 0.5f}, {0,0,1}}, {{ 0.5f,-0.5f,-0.5f}, {0,0,1}},
    {{ 0.5f, 0.5f,-0.5f}, {0,0,1}}, {{ 0.5f, 0.5f, 0.5f}, {0,0,1}},
    {{-0.5f,-0.5f,-0.5f}, {1,1,0}}, {{-0.5f,-0.5f, 0.5f}, {1,1,0}},
    {{-0.5f, 0.5f, 0.5f}, {1,1,0}}, {{-0.5f, 0.5f,-0.5f}, {1,1,0}},
    {{-0.5f, 0.5f, 0.5f}, {0,1,1}}, {{ 0.5f, 0.5f, 0.5f}, {0,1,1}},
    {{ 0.5f, 0.5f,-0.5f}, {0,1,1}}, {{-0.5f, 0.5f,-0.5f}, {0,1,1}},
    {{-0.5f,-0.5f,-0.5f}, {1,0,1}}, {{ 0.5f,-0.5f,-0.5f}, {1,0,1}},
    {{ 0.5f,-0.5f, 0.5f}, {1,0,1}}, {{-0.5f,-0.5f, 0.5f}, {1,0,1}}
};

static const uint16_t kIndices[36] = {
     0, 1, 2,  0, 2, 3,   4, 5, 6,  4, 6, 7,
     8, 9,10,  8,10,11,  12,13,14, 12,14,15,
    16,17,18, 16,18,19,  20,21,22, 20,22,23
};

static const char *kShaderSrc =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct V { float3 p; float3 c; };\n"
"struct U { float4x4 mvp; };\n"
"struct O { float4 pos [[position]]; float3 col; };\n"
"vertex O vs(const device V* v [[buffer(0)]], constant U& u [[buffer(1)]], uint id [[vertex_id]]) {\n"
"  O o;\n"
"  o.pos = u.mvp * float4(v[id].p, 1.0);\n"
"  o.col = v[id].c;\n"
"  return o;\n"
"}\n"
"fragment float4 fs(O in [[stage_in]]) { return float4(in.col, 1.0); }\n";

static inline simd_float4x4 Persp(float fovy, float aspect, float n, float f) {
    float ys = 1.0f / tanf(fovy * 0.5f);
    float xs = ys / aspect;
    float zs = f / (n - f);
    return simd_matrix(
        simd_make_float4(xs, 0, 0, 0),
        simd_make_float4(0, ys, 0, 0),
        simd_make_float4(0, 0, zs, -1),
        simd_make_float4(0, 0, zs * n, 0));
}

static inline simd_float4x4 LookAt(simd_float3 eye, simd_float3 center, simd_float3 up) {
    simd_float3 z = simd_normalize(eye - center);
    simd_float3 x = simd_normalize(simd_cross(up, z));
    simd_float3 y = simd_cross(z, x);
    return simd_matrix(
        simd_make_float4(x.x, y.x, z.x, 0),
        simd_make_float4(x.y, y.y, z.y, 0),
        simd_make_float4(x.z, y.z, z.z, 0),
        simd_make_float4(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1));
}

static inline simd_float4x4 RotY(float r) {
    float c = cosf(r), s = sinf(r);
    return simd_matrix(
        simd_make_float4(c, 0, -s, 0),
        simd_make_float4(0, 1, 0, 0),
        simd_make_float4(s, 0, c, 0),
        simd_make_float4(0, 0, 0, 1));
}

@interface Renderer : NSObject <MTKViewDelegate>
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLRenderPipelineState> pso;
@property (nonatomic, strong) id<MTLDepthStencilState> ds;
@property (nonatomic, strong) id<MTLBuffer> vb;
@property (nonatomic, strong) id<MTLBuffer> ib;
@property (nonatomic, strong) id<MTLBuffer> ub;
@property (nonatomic, assign) float t;
@end

@implementation Renderer
- (instancetype)initWithView:(MTKView *)view {
    if (self = [super init]) {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) { NSLog(@"no metal device"); return self; }
        view.device = _device;
        view.delegate = self;
        view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
        view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
        view.clearColor = MTLClearColorMake(0.05, 0.05, 0.08, 1.0);
        view.clearDepth = 1.0;
        _queue = [_device newCommandQueue];

        NSError *e = nil;
        id<MTLLibrary> lib = [_device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSrc]
                                                   options:nil error:&e];
        if (!lib) { NSLog(@"shader error: %@", e); return self; }

        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"vs"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
        pd.colorAttachments[0].pixelFormat = view.colorPixelFormat;
        pd.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
        _pso = [_device newRenderPipelineStateWithDescriptor:pd error:&e];
        if (!_pso) { NSLog(@"pipeline error: %@", e); return self; }

        MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
        dd.depthCompareFunction = MTLCompareFunctionLess;
        dd.depthWriteEnabled = YES;
        _ds = [_device newDepthStencilStateWithDescriptor:dd];

        _vb = [_device newBufferWithBytes:kVertices length:sizeof(kVertices) options:MTLResourceStorageModeShared];
        _ib = [_device newBufferWithBytes:kIndices length:sizeof(kIndices) options:MTLResourceStorageModeShared];
        _ub = [_device newBufferWithLength:sizeof(Uniforms) options:MTLResourceStorageModeShared];
    }
    return self;
}

- (void)drawInMTKView:(MTKView *)view {
    if (!_pso || !view.currentDrawable) return;
    _t += 1.0f / 60.0f;
    CGSize sz = view.drawableSize;
    float aspect = (float)sz.width / (float)sz.height;

    Uniforms u;
    u.mvp = simd_mul(Persp(60.0f * (float)M_PI / 180.0f, aspect, 0.1f, 100.0f),
                     simd_mul(LookAt(simd_make_float3(2.5f, 2.0f, 2.5f),
                                     simd_make_float3(0, 0, 0),
                                     simd_make_float3(0, 1, 0)),
                              RotY(_t)));
    memcpy(_ub.contents, &u, sizeof(u));

    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    id<MTLCommandBuffer> cb = [_queue commandBuffer];
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

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {}
@end

@interface VC : UIViewController
@property (nonatomic, strong) Renderer *r;
@end

@implementation VC
- (void)viewDidLoad {
    [super viewDidLoad];
    MTKView *mv = [[MTKView alloc] initWithFrame:self.view.bounds];
    mv.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    mv.preferredFramesPerSecond = 60;
    [self.view addSubview:mv];
    self.r = [[Renderer alloc] initWithView:mv];
}
@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([VC class]));
    }
}
