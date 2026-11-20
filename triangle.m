#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <simd/simd.h>

typedef struct {
    simd_float3 pos;
    simd_float3 col;
} Vertex;

@interface Renderer : NSObject <MTKViewDelegate>
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLRenderPipelineState> pso;
@property (nonatomic, strong) id<MTLDepthStencilState> ds;
@property (nonatomic, strong) id<MTLBuffer> vb;
@property (nonatomic, strong) id<MTLBuffer> ib;
@property (nonatomic, strong) id<MTLBuffer> ub;
@end

@implementation Renderer
- (instancetype)initWithView:(MTKView *)view {
    self = [super init];
    if (!self) return nil;
    
    _device = MTLCreateSystemDefaultDevice();
    view.device = _device;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    view.delegate = self;
    
    _queue = [_device newCommandQueue];
    
    NSError *err = nil;
    id<MTLLibrary> lib = [_device newDefaultLibrary];
    if (!lib) {
        NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\nstruct V2F { float4 pos [[position]]; float3 col; };\nvertex V2F vs(V2F in [[stage_in]]) { V2F o; o.pos = float4(in.pos, 1.0); o.col = in.col; return o; }\ndragment float4 fs(V2F in [[stage_in]]) { return float4(in.col, 1.0); }";
        lib = [_device newLibraryWithSource:src options:nil error:&err];
    }
    id<MTLFunction> vs = [lib newFunctionWithName:@"vs"];
    id<MTLFunction> fs = [lib newFunctionWithName:@"fs"];
    
    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    pd.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    pd.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
    _pso = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
    
    MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled = YES;
    _ds = [_device newDepthStencilStateWithDescriptor:dd];
    
    Vertex v[] = {
        {{-0.5,-0.5,0.5},{1,0,0}}, {{{0.5,-0.5,0.5},{0,1,0}}, {{{0.5,0.5,0.5},{0,0,1}}, {{-0.5,0.5,0.5},{1,1,0}},
        {{-0.5,-0.5,-0.5},{1,0,1}}, {{{0.5,-0.5,-0.5},{0,1,1}}, {{{0.5,0.5,-0.5},{1,0,0}}, {{-0.5,0.5,-0.5},{0,1,0}}
    };
    uint16_t i[] = {
        0,1,2, 0,2,3,  4,6,5, 4,7,6,
        0,4,7, 0,7,3,  1,5,6, 1,6,2,
        3,7,6, 3,6,2,  0,1,5, 0,5,4
    };
    _vb = [_device newBufferWithBytes:v length:sizeof(v) options:MTLResourceStorageModeShared];
    _ib = [_device newBufferWithBytes:i length:sizeof(i) options:MTLResourceStorageModeShared];
    _ub = [_device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    
    return self;
}

- (void)drawInMTKView:(MTKView *)view {
    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    if (!rp) return;
    
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.1, 0.1, 1.0);
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:_pso];
    [enc setDepthStencilState:_ds];
    [enc setCullMode:MTLCullModeBack];
    [enc setFrontFacingWinding:MTLWindingCounterClockwise];
    [enc setVertexBuffer:_vb offset:0 atIndex:0];
    [enc setVertexBuffer:_ub offset:0 atIndex:1];
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:36 indexType:MTLIndexTypeUInt16 indexBuffer:_ib indexBufferOffset:0];
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
