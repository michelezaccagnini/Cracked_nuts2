//MIDI data parsing

#define FEEDBACK iChannel0
#define MIDI iChannel1
#define BUF_A iChannel2

#define HI_CHAN 5
#define NUM_HI_1 3
#define HI_ENV_ROW 0

const int[NUM_HI_1] hi_pitch = int[NUM_HI_1](96,86,70);
const ivec3[NUM_HI_1] cc_ind = 
        ivec3[NUM_HI_1](ivec3(20,21,22),ivec3(23,24,25),ivec3(26,27,28));

const vec3[NUM_HI_1] offs = vec3[NUM_HI_1](
    vec3(-0.2,0.2,0),vec3(0,-0.2,0),vec3(0.2,0,0)
);
vec4[NUM_HI_1] HI_POS;

//debug: visualize midi texture
#if 0
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragColor = texelFetch(MIDI,ivec2(fragCoord),0);
}
#else

// void SetVertex(sampler2D midi)
// {
//     for(int id = 0; id < NUM_HI_1;id++)
//     {
//         ivec3 cc = cc_ind[id];
//         vec3 off = offs[id];
//         vec3 pos = vec3(getCCval(cc.x,HI_CHAN,midi),
//                         getCCval(cc.y,HI_CHAN,midi),
//                         getCCval(cc.z,HI_CHAN,midi));

//         float env = texelFetch(FEEDBACK,ivec2(id,HI_ENV_ROW),0).x;
//         HI_POS[id] = vec4(pos*1.3-0.5+off,env);
//     }
// }

vec3 Dot_pos(int id, sampler2D midi)
{
    ivec3 cc_vec;
    if(id == 0) cc_vec = ivec3(1,2,3);
    if(id == 1) cc_vec = ivec3(4,5,6);
    vec3 pos = vec3(getCCval(cc_vec.x,HI_CHAN,midi),
                    getCCval(cc_vec.y,HI_CHAN,midi),
                    getCCval(cc_vec.z,HI_CHAN,midi));

    return pos;
}
vec4 getEnvelopeHi(int id, int chan, int row, float dur, float sli, sampler2D midi, sampler2D feedb)
{
    int pitch = hi_pitch[id];
    vec4  prev_data = iFrame < 10 ? vec4(0) : texelFetch(feedb,ivec2(id,row),0);
    float prev_env  = prev_data.x, 
            was_on  = prev_data.y;
    bool on = false;
    float vel = vel_note_on(midi, chan, pitch, on);
    float is_on = on ? 1. : 0.;
    bool trigger = on && was_on < 0.5;
    float env = 0.;
    if(trigger) 
    {
        //adapt to heard velocity instead of midi vel?
        env = vel;//*0.2+0.5;
    }
    else 
    {
        env = slide(prev_env, 0., 1./dur); 
    }
    return vec4(slide(prev_env,env,sli), is_on, pitch,0);
}

float Dot(vec2 uv, vec4 p)
{
    float d = length(uv-p.xy)-(p.z*0.2);
    return d;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    ivec2 iCoo = ivec2(fragCoord);
    //if(all(lessThan(iCoo,ivec2(NUM_HI+1))))
    if(iCoo.y == 0)
    {
        if(iCoo.x < NUM_HI_1 )fragColor = getEnvelopeHi(iCoo.x,HI_CHAN,HI_ENV_ROW, 3.,0.9,MIDI,FEEDBACK);
        else discard;
    }
    else if(iCoo.y <= NUM_HI_1)
    {
        if(iCoo.x == 0)
        {
            const float mult = 0.8;
            int id= iCoo.y-1;
            float env = texelFetch(FEEDBACK,ivec2(id,0),0).x;
            float tar = texelFetch(FEEDBACK,iCoo,0).y+env*mult;
            float cur = texelFetch(FEEDBACK,iCoo,0).y;
            float acc = slide(cur,tar,0.8);
            fragColor = vec4(env,acc,0,0);
        }
        else if(iCoo.x < POS_HISTORY) fragColor = pix_stream4(iCoo,FEEDBACK,0.4);
        else discard;
    }
    else
    {
        vec2 uv =2.*(fragCoord-iResolution.xy*0.5)/iResolution.y;
        // float kick = pow(texelFetch(BUF_A,ivec2(3,DRUMS_ENV_ROW),0).x,2.5);
        vec3 col = vec3(0);
        //SetVertex(MIDI);
        // float cap1 = sdUnevenCapsule(uv,HI_POS[0].xy,HI_POS[1].xy,HI_POS[0].w*0.2+0.1,HI_POS[1].w*0.2+0.1);//Dot(uv,HI_POS[0]+vec4(0,0-0.5,0,0));////Dot(uv,HI_POS[0].xyz);//
        // float cap2 = sdUnevenCapsule(uv,HI_POS[1].xy,HI_POS[2].xy,HI_POS[1].w*0.2+0.1,HI_POS[2].w*0.2+0.1);//Dot(uv,HI_POS[1]+vec4(0.5,-1.,0,0));////Dot(uv,HI_POS[1].xyz);//
        // float cap3 = sdUnevenCapsule(uv,HI_POS[2].xy,HI_POS[0].xy,HI_POS[2].w*0.2+0.1,HI_POS[0].w*0.2+0.1);//Dot(uv,HI_POS[2]+vec4(-0.5,0,0,0));// //Dot(uv,HI_POS[2].xyz);//
        vec3 dot_pos1 = Dot_pos(0,MIDI);
        vec3 dot_pos2 = Dot_pos(1,MIDI)*vec3(-1,-1,1);
        float Dot1 = length(uv-dot_pos1.xy)-(dot_pos1.z*0.2+0.1);
        float Dot2 = length(uv-dot_pos2.xy)-(dot_pos2.z*0.2+0.1);
        float Dot = min(Dot1,Dot2);
        
        col = Dot < 0. ? vec3(1) : vec3(0);
        vec3 feed = texture(FEEDBACK,fragCoord/iResolution.xy).xyz;
        vec3 col2  = mix(col,feed,0.4);
        
        fragColor = vec4(col2,0);

    }
    

}
#endif