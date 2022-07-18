#define BUF_A iChannel0
#define MIDI iChannel1
#define BUF_B iChannel2
#define BUF_C iChannel3
#define ZERO (min(iFrame,0))

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    

    vec3 col = false? texelFetch(BUF_C,ivec2(fragCoord*vec2(0.05, 0.1)),0).xyz : false ? 
                                                                                 texelFetch(BUF_C,ivec2(fragCoord),0).xyz : 
                                                                                 true?
                                                                                 texelFetch(BUF_B,ivec2(fragCoord*vec2(1.)),0).xyz :
                                                                                 texelFetch(BUF_C,ivec2(fragCoord*vec2(1.)),0).xyz ; 

    col = pow(max(col,vec3(0)),vec3(0.4545));
    fragColor = vec4(col,1.0);

}