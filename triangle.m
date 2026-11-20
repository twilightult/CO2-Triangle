#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface Renderer : NSObject <MTKViewDelegate>
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLRenderPipelineState> pso;
@property (nonatomic, strong) id<MTLBuffer> vb;
@end

@implementation Renderer
- (instancetype)initWithView:(MTKView *)view {
    self = [super init];
    if (!self) return nil;
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) return self;
    view.device = _device;
    view.delegate = self;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.clearColor = MTLClearColorMake(0.1, 0.2, 0.3, 1.0);
    _queue = [_device newCommandQueue];

    const char *src =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct V { float2 p; float3 c; };\n"
    "struct O { float4 pos [[position]]; float3 col; };\n"
    "vertex O vs(const device V* v [[buffer(0)]], uint id [[vertex_id]]) {\n"
    "    O o; o.pos = float4(v[id].p, 0.0, 1.0); o.col = v[id].c; return o;\n"
    "}\n"
    "fragment float4 fs(O in [[stage_in]]) { return float4(in.col, 1.0); }\n";

    NSError *e = nil;
    id<MTLLibrary> lib = [_device newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&e];
    if (!lib) { NSLog(@"shader: %@", e); return self; }
    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [lib newFunctionWithName:@"vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
    pd.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    _pso = [_device newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!_pso) { NSLog(@"pso: %@", e); return self; }

    static const float verts[] = {
         0.0f,  0.5f,  1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f,  0.0f, 1.0f, 0.0f,
         0.5f, -0.5f,  0.0f, 0.0f, 1.0f
    };
    _vb = [_device newBufferWithBytes:verts length:sizeof(verts) options:MTLResourceStorageModeShared];
    return self;
}
- (void)drawInMTKView:(MTKView *)view {
    if (!_pso || !view.currentDrawable) return;
    MTLRenderPassDescriptor *rp = view.currentRenderPassDescriptor;
    id<MTLCommandBuffer> cb = [_queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:_pso];
    [enc setVertexBuffer:_vb offset:0 atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
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
    [self.view addSubview:mv];
    self.r = [[Renderer alloc] initWithView:mv];
}
@end

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (nonatomic, strong) UIWindow *window;
@end
@implementation AppDelegate
- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    self.window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];
    self.window.rootViewController = [[VC alloc] init];
    [self.window makeKeyAndVisible];
    return YES;
}
@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
    }
}
