#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>

const char* shaderSrc = R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 v_main(uint vid [[vertex_id]]) {
    float2 p[] = { {-0.5, -0.5}, {0.5, -0.5}, {0.0, 0.5} };
    return float4(p[vid], 0.0, 1.0);
}
fragment float4 f_main() { return float4(0.0, 1.0, 1.0, 1.0); }
)";

@interface Renderer : NSObject <MTKViewDelegate>
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) id<MTLRenderPipelineState> pipeline;
@end

@implementation Renderer
- (instancetype)initWithView:(MTKView *)view {
    self = [super init];
    _device = view.device;
    _queue = [_device newCommandQueue];
    id<MTLLibrary> lib = [_device newLibraryWithSource:[NSString stringWithUTF8String:shaderSrc] options:nil error:nil];
    MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [lib newFunctionWithName:@"v_main"];
    desc.fragmentFunction = [lib newFunctionWithName:@"f_main"];
    desc.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    _pipeline = [_device newRenderPipelineStateWithDescriptor:desc error:nil];
    return self;
}
- (void)drawInMTKView:(MTKView *)view {
    id<MTLCommandBuffer> cmd = [_queue commandBuffer];
    MTLRenderPassDescriptor *pass = view.currentRenderPassDescriptor;
    if (pass) {
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];
        [enc setRenderPipelineState:_pipeline];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cmd presentDrawable:view.currentDrawable];
    }
    [cmd commit];
}
- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {}
@end

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property (strong, nonatomic) UIWindow *window;
@property (strong, nonatomic) Renderer *renderer;
@property (strong, nonatomic) MTKView *metalView;
@end

@implementation AppDelegate
- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    UIViewController *vc = [[UIViewController alloc] init];
    self.window.rootViewController = vc;
    [self.window makeKeyAndVisible];
    self.metalView = [[MTKView alloc] initWithFrame:self.window.bounds];
    self.metalView.device = MTLCreateSystemDefaultDevice();
    self.metalView.clearColor = MTLClearColorMake(0.1, 0.1, 0.1, 1.0);
    self.renderer = [[Renderer alloc] initWithView:self.metalView];
    self.metalView.delegate = self.renderer;
    [vc.view addSubview:self.metalView];
    return YES;
}
@end

int main(int argc, char * argv[]) {
    @autoreleasepool { return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class])); }
}
