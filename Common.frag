

#define test
#define TIME_UNIT 0.18 // 180 ms = 83.33 bpm
#define FPS 60
const float STREAM_SLIDE = FPS == 60 ? 0.5 : 1.;  
#define MAX_DIST 200.
#define LOOKAT vec3(cos(iTime),sin(iTime),0)*0.8
#define TAU 6.28318530718
#define PI 3.14159265359
#define RHO 1.570796326795
const float POS_SLIDE_AMT = 0.9;

const float ENV_SLIDE_AMT = 0.5; 
//#define BACK_PLANE

#define AA 1


#define POS_HISTORY 30
const float fPOS_HISTORY = float(POS_HISTORY);

//rope uniforms
#define ROPE_POINTS 7
const float fROPE_POINTS = float(ROPE_POINTS);
#define ENV_ROW 0

const int FIRST_BLOCK_SIZE = 1;

/*DRUMS BLOCK*/
#define DRUMS_CHAN 0
const int NUM_DRUMS_COMBINATIONS  = 6;
#define NUM_DRUMS 4
#define DRUMS_ENV_ROW 0
#define NUM_ENV_ROW 1
const int DRUMS_BLOCK_SIZE =  NUM_DRUMS + NUM_ENV_ROW;
const int DRUMS_TOT_BLOCK_SIZE = DRUMS_BLOCK_SIZE + NUM_DRUMS_COMBINATIONS;
const int[NUM_DRUMS] drum_pitch = int[NUM_DRUMS](55,51,65,39);
const float[NUM_DRUMS] drum_dur = float[NUM_DRUMS](4,4,4,8);
const float[NUM_DRUMS] drum_vel = float[NUM_DRUMS](1,1,1,1);

const float DRUM_THICK = 1.2;
const float DRUM_Z_DISPL = 30.;

vec3[NUM_DRUMS] DRUMS_VERTEX_POS;

/*MID BLOCK*/
#define MID_CHAN 2
#define NUM_MID 1
const int MID_ENV_ROW = DRUMS_TOT_BLOCK_SIZE;
const int MID_BLOCK_SIZE = NUM_MID + NUM_ENV_ROW;
const int MID_TOT_BLOCK_SIZE = MID_BLOCK_SIZE;
const int MID_BLOCK_OFFSET = DRUMS_TOT_BLOCK_SIZE;
const float MID_Z_DISPL = 25.;

/*MID-HI BLOCK*/
#define MID_HI_CHAN 3
#define NUM_MID_HI 3    
const int[NUM_MID_HI] mid_hi_pitch = int[NUM_MID_HI](60,82,83);
const int MID_HI_ENV_ROW = DRUMS_TOT_BLOCK_SIZE + MID_TOT_BLOCK_SIZE;
const int MID_HI_BLOCK_SIZE = NUM_MID_HI + NUM_ENV_ROW;
const int MID_HI_BLOCK_OFFSET = MID_TOT_BLOCK_SIZE + DRUMS_TOT_BLOCK_SIZE;
const int MID_HI_TOT_BLOCK_SIZE = MID_HI_BLOCK_SIZE;
const float MID_HI_RAD = 0.8;

const float MID_HI_Z_DISPL = 40.;

/*BASS BLOCK*/
#define BASS_CHAN 1
#define NUM_BASS 4  
const int NUM_BASS_COMBINATIONS  = 6;
//bass pitch array is in getEnvelopeGivenPitch()
const int BASS_ENV_ROW = DRUMS_TOT_BLOCK_SIZE + MID_TOT_BLOCK_SIZE + MID_HI_TOT_BLOCK_SIZE;
const int BASS_BLOCK_SIZE = NUM_BASS + NUM_ENV_ROW;
const int BASS_BLOCK_OFFSET = MID_TOT_BLOCK_SIZE + DRUMS_TOT_BLOCK_SIZE + MID_HI_TOT_BLOCK_SIZE;
const int BASS_TOT_BLOCK_SIZE = BASS_BLOCK_SIZE + NUM_BASS_COMBINATIONS;
vec3[NUM_BASS] BASS_VERTEX_POS;

const float BASS_THICK = 0.9;

const float BASS_Z_DISPL = 30.;




const vec3 POS_PLANE_Y = vec3(0,-15.5,0);
const vec3 POS_PLANE_Z = vec3(0,0,-5);

const float sph_smin = 0.5;

//Ray origin
#define RAY_OR_CHAN 15
#define RAY_OR_SPAN 30.

#define LOOKAT_SPAN 20.



const vec3[NUM_DRUMS] DRUMS_FIX_POS = vec3[NUM_DRUMS]
(
    // vec3( 5,-5,-5),//a
    // vec3(-5, 5,-5),//b
    // vec3( 5,-5, 5),//c
    // vec3( 5, 5, 5) //d
    vec3(-4, 4,-4),
    vec3(-4,-4, 4),
    vec3( 4, 4, 4),
    vec3( 4,-4,-4) 
);

const ivec2[NUM_DRUMS_COMBINATIONS] DRUMS_COMBINATIONS = ivec2[NUM_DRUMS_COMBINATIONS]
                    (ivec2(1,2),
                    ivec2(1,3),
                    ivec2(1,4),
                    ivec2(2,3),
                    ivec2(2,4),
                    ivec2(3,4)
                    );

const vec3[NUM_BASS] BASS_FIX_POS = vec3[NUM_BASS]
(
    // vec3(-5, 5,-5),//e
    // vec3( 5, 5, 5),//f
    // vec3( 5,-5, 5),//g
    // vec3(-5, 5, 5) //h
     vec3(-4, 4,-4),
    vec3(-4,-4, 4),
    vec3( 4, 4, 4),
    vec3( 4,-4,-4)  
);

const ivec2[NUM_BASS_COMBINATIONS] BASS_COMBINATIONS = ivec2[NUM_BASS_COMBINATIONS]
                    (ivec2(1,2),
                    ivec2(1,3),
                    ivec2(1,4),
                    ivec2(2,3),
                    ivec2(2,4),
                    ivec2(3,4)
                    );


mat3 camera( in vec3 ro, in vec3 ta, float cr )
{
	vec3 cw = normalize(ta-ro);
	vec3 cp = vec3(sin(cr), cos(cr),0.0);
	vec3 cu = normalize( cross(cw,cp) );
	vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

float slide(float cur, float tar, float sli)
{
    return mix(cur, tar, sli);
}
vec2 slide(vec2 cur, vec2 tar, float sli)
{ 
    return mix(cur, tar, sli);
}
vec3 slide(vec3 cur, vec3 tar, float sli)
{ 
    return mix(cur, tar, sli);
}
vec4 slide(vec4 cur, vec4 tar, float sli)
{ 
    return mix(cur, tar, sli);
}

float ease(float x){return x*x*(3.-2.*x);}

float bounce(float x, float r)
{
    x = fract(-log(1.-x)/log(r));
    float b = -x*(x-1.);
    return b;
}

// converts two polar angles uv from texture to point U on a sphere with radius 1
vec3 pol2Car(vec2 uv)
{
    uv = vec2(2, 1) * (uv - .5) * PI;
    return vec3(sin(uv.x), 0, cos(uv.x)) * cos(uv.y) + vec3(0, sin(uv.y), 0);
}

vec3 carToPol(vec3 p) {
    float r = length(p);
    float the = acos(p.z/r);
    float phi = atan(p.y,p.x);
    return vec3(r,the,phi);
}

vec3 pix_stream(ivec2 coo, sampler2D text, float sli)
{
    vec3 tar = texelFetch(text, coo - ivec2(1,0),0).xyz;
    vec3 cur = texelFetch(text, coo ,0).xyz;
    return  slide(cur,tar,sli);
}

vec4 pix_stream4(ivec2 coo, sampler2D text, float sli)
{
    vec4 tar = texelFetch(text, coo - ivec2(1,0),0);
    vec4 cur = texelFetch(text, coo ,0);
    return  slide(cur,tar,sli);
}

float hash12(vec2 p)
{
	p  = fract(p * .1031);
    p += dot(p, p.yx + 33.33);
    return fract(p.x  * p.y);
}

vec3 opU(in vec3 a, in vec3 b)
{
    return a.x < b.x ? a : b; 
}

float tri(float x)
{
    return min(fract(x) * 2., 2. - 2. * fract(x));
}

vec3 hash31( float n )
{
    return fract(sin(vec3(n,n+1.0,n+2.0))*vec3(43758.5453123,22578.1459123,19642.3490423));
}

mat2 rotate(float ang)
{
    return mat2(cos(ang), sin(ang),-sin(ang), cos(ang));
}

float smax( in float a, in float b, in float s ){
    float h = clamp( 0.5 + 0.5*(a-b)/s, 0.0, 1.0 );
    return mix(b, a, h) + h*(1.0-h)*s;
}
vec2 to_polar(vec3 U)
{
   return fract(vec2(
        atan(U.z, U.x) / 2.,
        atan(U.y, length(U.xz))
    ) / PI + .5);
}

vec3 to_cartesian(vec2 uv)
{
    uv = vec2(2, 1) * (uv - .5) * PI;
    return vec3(cos(uv.x), 0, sin(uv.x)) * cos(uv.y) + vec3(0, sin(uv.y), 0);
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r)
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}


float cro(in vec2 a, in vec2 b ) { return a.x*b.y - a.y*b.x; }

float sdUnevenCapsule( in vec2 p, in vec2 pa, in vec2 pb, in float ra, in float rb )
{
    p  -= pa;
    pb -= pa;
    float h = dot(pb,pb);
    vec2  q = vec2( dot(p,vec2(pb.y,-pb.x)), dot(p,pb) )/h;
    
    //-----------
    
    q.x = abs(q.x);
    
    float b = ra-rb;
    vec2  c = vec2(sqrt(h-b*b),b);
    
    float k = cro(c,q);
    float m = dot(c,q);
    float n = dot(q,q);
    
         if( k < 0.0 ) return sqrt(h*(n            )) - ra;
    else if( k > c.x ) return sqrt(h*(n+1.0-2.0*q.y)) - rb;
                       return m                       - ra;
}

vec2 polar(vec3 U)
{
   return fract(vec2(
        atan(U.z, U.x) / 2.,
        atan(U.y, length(U.xz))
    ) / PI + .5);
}
//==========================================================================================
//MIDI
//==========================================================================================
#define HEIGHT_CH_BLOCK 5

float getCCval (int CC, int channel,sampler2D midi_data)
{
    //add 1 to the channel to make it start from 1
    channel += 1;
    int   yco = (channel)*HEIGHT_CH_BLOCK-2;
    ivec2 coo = ivec2(CC, yco);
    float ccv = texelFetch(midi_data, coo,0).x;
    return ccv;
}


//==========================================================================================
//ONESHADE code
//==========================================================================================
//https://www.shadertoy.com/view/slXXD4
// Definite integral of the light volume over the entire view of the ray (0 to ∞)
// I computed the limit based on the fact that atan() approaches π/2 (90 degrees)
// as x (the ratio of y over x) goes to infinity
float integrateLightFullView(in vec3 ro, in vec3 rd, in float k, in float d) 
{
    float a = dot(rd, rd);
    float b = dot(ro, rd);
    float c = dot(ro, ro) + d * d;
    float h = sqrt(a * c - b * b);
    return d * d * k * (RHO - atan(b / h)) / h;
}
//==========================================================================================
//NR4 code
//==========================================================================================

const vec3 c = vec3(1.,0.,-1.);
const float pi = acos(-1.);
// Determine zeros of k.x*x^2+k.y*x+k.z
vec2 quadratic_zeros(vec3 k)
{
    if(k.x == 0.) return -k.z/k.y*c.xx; // is a rare edgecase
    float d = k.y*k.y-4.*k.x*k.z;
    if(d<0.) return vec2(1.e4);
    return (c.xz*sqrt(d)-k.y)/(2.*k.x);
}
// Determine zeros of k.x*x^3+k.y*x^2+k.z*x+k.w
vec3 cubic_zeros(vec4 k)
{
    if(k.x == 0.) return quadratic_zeros(k.yzw).xyy;
    
    // Depress
    vec3 ai = k.yzw/k.x;
    
    //discriminant and helpers
    float tau = ai.x/3., 
        p = ai.y-tau*ai.x, 
        q = -tau*(tau*tau+p)+ai.z,
        dis = q*q/4.+p*p*p/27.;
        
    //triple real root
    if(dis > 0.) {
        vec2 ki = -.5*q*c.xx+sqrt(dis)*c.xz, 
            ui = sign(ki)*pow(abs(ki), c.xx/3.);
        return vec3(ui.x+ui.y-tau);
    }
    
    //three distinct real roots
    float fac = sqrt(-4./3.*p), 
        arg = acos(-.5*q*sqrt(-27./p/p/p))/3.;
    return c.zxz*fac*cos(arg*c.xxx+c*pi/3.)-tau;
}
// Point on a spline
vec3 xspline3(vec3 x, float t, vec3 p0, vec3 p1, vec3 p2)
{
    return mix(mix(p0,p1,t),mix(p1,p2,t),t);
}
// Distance to a point on a spline
float dspline3(vec3 x, float t, vec3 p0, vec3 p1, vec3 p2)
{
    return length(x - xspline3(x, t, p0, p1, p2));
}
// spline parameter of the point with minimum distance on the spline and sdf
// Returns vec2(dmin, tmin).
vec2 dtspline3(vec3 x, vec3 p0, vec3 p1, vec3 p2)
{
    vec3 E = x-p0, F = p2-2.*p1+p0, G = p1-p0;
    E = clamp(cubic_zeros(vec4(dot(F,F), 3.*dot(G,F), 2.*dot(G,G)-dot(E,F), -dot(E,G))),0.,1.);
    F = vec3(dspline3(x,E.x,p0,p1,p2),dspline3(x,E.y,p0,p1,p2),dspline3(x,E.z,p0,p1,p2));
    return F.x < F.y && F.x < F.z
        ? vec2(F.x, E.x)
        : F.y < F.x && F.y < F.z
            ? vec2(F.y, E.y)
            : vec2(F.z, E.z);
}
// Normal in a point on a spline
vec3 nspline3(vec3 x, float t, vec3 p0, vec3 p1, vec3 p2)
{
    return normalize(mix(p1-p0, p2-p1, t));
}
//distance to spline with parameter t
float dist(vec3 p0,vec3 p1,vec3 p2,vec3 x,float t)
{
    t = clamp(t, 0., 1.);
    return length(x-pow(1.-t,2.)*p0-2.*(1.-t)*t*p1-t*t*p2);
}
float dbox3(vec3 x, vec3 b)
{
  b = abs(x) - b;
  return length(max(b,0.))
         + min(max(b.x,max(b.y,b.z)),0.);
}
// Compute an orthonormal system from a single vector in R^3
mat3 ortho(vec3 d)
{
    vec3 a = normalize(
        d.x != 0. 
            ? vec3(-d.y/d.x,1.,0.)
            : d.y != 0.
                ? vec3(1.,-d.x/d.y,0.)
                : vec3(1.,0.,-d.x/d.z)
    );
    return mat3(d, a, cross(d,a));
}

vec2 asphere(vec3 x, vec3 dir, float R)
{
    float a = dot(dir,dir),
        b = 2.*dot(x,dir),
        cc = dot(x,x)-R*R,
        dis = b*b-4.*a*cc;
    if(dis<0.) return vec2(1000.);
    vec2 dd = (c.xz*sqrt(dis)-b)/2./a;
    return vec2(min(dd.x, dd.y), max(dd.x, dd.y));
}
/*
===================================================================================
*/

vec3 point_on_plane(vec3 ro, vec3 rd, vec3 p, vec3 norm)
{
    float t = max(0., dot(p-ro,norm)/dot(rd,norm));
    return ro+rd*t;
}

float sdPlane( vec3 p )
{
	return p.y;
}

float sdPlaneZ( vec3 p )
{
	return p.z;
}
float sdCappedCylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float smin( float a, float b, float k , inout float h ) 
{
    h = clamp( 0.5+0.5*(b-a)/k, 0., 1. );
    return mix( b, a, h ) - k*h*(1.0-h);
}

vec3 rope(vec3 p, int rope_id, sampler2D text, float env, float thick, inout vec3 hit_point)
{
    float dist = 1000.;
    vec2 uv;
    for(int i = 0; i < ROPE_POINTS-2; i+=2)
    {
        vec3 pp1  = texelFetch(text,ivec2(i+0,rope_id),0).xyz;
        vec3 pp2  = texelFetch(text,ivec2(i+1,rope_id),0).xyz;
        vec3 pp3  = texelFetch(text,ivec2(i+2,rope_id),0).xyz;
        if(length(pp1-pp2) < 0.001) return vec3(100);//avoid mapping static objects
        vec2 b = dtspline3(p,pp1,pp2,pp3);
        float t = b.y;
        vec3 norm  = nspline3(p,t,pp1,pp2,pp3);
        vec3 point = xspline3(p,t,pp1,pp2,pp3);
        mat3 m = ortho(norm),
            mt = transpose(m);
        vec3 c_point = mt*(p-point);
        
        const float span = 1./4.;
        float bez_ind = floor(float(i)/2.);
        vec2 lwise =vec2(t*span+bez_ind*span,t);
        float l = pow(lwise.x,0.7);
        float w =thick;//smoothstep(0.4,0.,abs(l-(1.-pow(env,0.5))))*1.8*smoothstep(0.8,0.6, l)*0.05+0.051;
        //float l = smoothstep(0.2,0.7,lwise.x)+0.01;
        
        //float d = dbox3(c_point, vec3(.1, w, w));
        float d = sdCappedCylinder(c_point,w,w);
        d = max(d,length(c_point)-w*0.9);
        //float d = sdCapsule(p,c_point,c_point*1.1,w);
        if(d < dist) 
        {
            hit_point = c_point;
            uv = vec2(c_point.z*w+w*0.5,lwise.x);
        }
        float hh;   
        dist =  min(dist,d);//smin(dist,d, 0.18);   
    }

    //uv not returned(add code)
    return vec3(dist, uv);
}



//==================================================================================================
//Animations
//==================================================================================================

float vel_note_on(sampler2D midi, int channel, int pitch, inout bool on) 
{
    ivec2 p = ivec2(pitch, HEIGHT_CH_BLOCK * channel );
    float vel = texelFetch(midi, p, 0).x;
    float secs_since_note_on  = texelFetch(midi, p + ivec2(0, 1), 0).x;
    float secs_since_note_off = texelFetch(midi, p + ivec2(0, 2), 0).x;
    float env = 0.;
    on = secs_since_note_on < secs_since_note_off || secs_since_note_off < 0.;
    return on ? vel : 0.;
}

int get_time_sector(float time, float width)
{
    int time_section = int(mod(time,width*TIME_UNIT)/TIME_UNIT);
    return time_section;
}
//cc list of parameters for RO and Lookat:
/*
10 : ro rotation angle on xy
11 : ro radius span 1
12 : ro radius span 2
13: lookat z (left right)
14 : lookat y (up down)
15 : lookat not assigned
*/
vec3 getRO(sampler2D midi)
{
    float x = getCCval(10,RAY_OR_CHAN,midi),
          y = getCCval(11,RAY_OR_CHAN,midi),
          z = getCCval(12,RAY_OR_CHAN,midi);


    vec3 pos = vec3(sin(x*TAU),cos(x*TAU),0);
    pos *= (RAY_OR_SPAN*z+RAY_OR_SPAN* y);
    return pos;
}

vec3 getLookat(float time, float kick, sampler2D midi)
{
    float x = getCCval(13,RAY_OR_CHAN,midi),
          y = getCCval(14,RAY_OR_CHAN,midi),
          z = getCCval(15,RAY_OR_CHAN,midi);

    float t1 = fract(time*3005.8)*TAU;
    float t2 = fract(time*2500.2)*TAU;

    vec3 pos = vec3(0,y,x);
    pos -= vec3(0,cos(t1),cos(t2))*(0.004+kick*0.5);
    return pos*LOOKAT_SPAN;
}
vec3 getEnvelope(int id, int chan, int row, float dur, float sli, sampler2D midi, sampler2D feedb)
{
    int pitch = int(getCCval(19,chan,midi)*127.+0.01) ;
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
        env = vel;
    }
    else 
    {
        env = slide(prev_env, 0., 1./dur); 
    }
    return vec3(slide(prev_env,env,sli), is_on, pitch);
}

vec3 getEnvelopeMidHi(int id, int chan, int row, float dur, float sli, sampler2D midi, sampler2D feedb)
{
    int pitch = mid_hi_pitch[id];
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
        env = vel;
    }
    else 
    {
        env = slide(prev_env, 0., 1./dur); 
    }
    return vec3(slide(prev_env,env,sli), is_on, pitch);
}

vec3 getEnvelopeGivenPitch(int id, int chan, int row, float dur, float sli, sampler2D midi, sampler2D feedb)
{
    int pitch = 0;
    if(chan == MID_HI_CHAN )    pitch = mid_hi_pitch[id];
    else if(chan == BASS_CHAN )
    {
        int[NUM_BASS] bass_pitch;
        bool first_bass = getCCval(15,BASS_CHAN,midi) < 0.5;
        if(first_bass) bass_pitch = int[NUM_BASS](39,55,65,56);
        else bass_pitch = int[NUM_BASS](63,67,48,41);
        pitch = bass_pitch[id];
    } 
    
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
        env = chan == BASS_CHAN ? 0.5 : vel;
    }
    else 
    {
        env = slide(prev_env, 0., 1./dur); 
    }
    return vec3(slide(prev_env,env,sli), is_on, vel);
}


//from https://suricrasia.online/demoscene/functions/
vec3 erot(vec3 p, vec3 ax, float ro) {
  return mix(dot(ax, p)*ax, p, cos(ro)) + cross(ax,p)*sin(ro);
}

vec3 z_appear(vec3 pos, float amt, int chan, sampler2D midi)
{
    //add z displacement to appear/disappear based on song section
    //CC 16 is 0 when is off, 1 when on
    float z_appear = amt-getCCval(16,chan,midi)*amt;
    pos.z += z_appear;
    return pos;
}

float get_CC_size(int chan, sampler2D midi)
{
    //size info for object is saved on CC 17
    return getCCval(17,chan,midi)*3.;
}
struct Note
{
    float env;
    float pitch;
    float vel;
    //time sector based on TIME_UNIT and width param
    float sector;
};


vec4 getEnvelopeDrums(int id, int chan, int row, float dur, float sli, float width, sampler2D midi, sampler2D feedb)
{
    
    int pitch = drum_pitch[id] ;
    float fix_vel = drum_vel[id];
    vec4  prev_data = iFrame < 10 ? vec4(0) : texelFetch(feedb,ivec2(id,row),0);
    float prev_env  = prev_data.x, 
            was_on  = prev_data.y,
            sector = prev_data.w;
    bool on = false;
    float vel = vel_note_on(midi, chan, pitch, on)*fix_vel;
    float is_on = on ? 1. : 0.;
    bool trigger = on && was_on < 0.5;
    float env = 0.;
    if(trigger) 
    {
        env = vel;
        sector = float(get_time_sector(iTime, width));
    }
    else 
    {
        env = slide(prev_env, 0., 1./dur); 
    }
    return vec4(slide(prev_env,env,sli), is_on, vel, sector);
}

vec3 getPosAnimDrums(int id, float amt, sampler2D feedb)
{
    vec3 pos = DRUMS_VERTEX_POS[id];
    for(int i = 0; i < NUM_DRUMS; i++)
    {
        if(id == i) continue;
        float env = texelFetch(feedb,ivec2(i,0),0).x;
        pos += DRUMS_VERTEX_POS[i]*env*amt; 
        //pos.xz *= rotate(tri(mod(iTime*0.001,0.01)));  
    }
    return pos;
}

vec4 animDataDrums(ivec2 tex_coo, sampler2D midi, sampler2D feedb)
{
    if(tex_coo.y == 0)
    {
        int id = tex_coo.x;
        int chan = 0;
        int row = DRUMS_ENV_ROW;
        float dur = drum_dur[id];
        float width = 4.;
        return getEnvelopeDrums(id,chan, row,dur,ENV_SLIDE_AMT,width, midi,feedb);
    }
    else if(tex_coo.y < DRUMS_BLOCK_SIZE)
    {
        if (tex_coo.x == 0)
        {
            int id = tex_coo.y-1;
            //attach each stem to the center
            float anim_amt = 0.6;
            vec3 pos = getPosAnimDrums(id,anim_amt,feedb);

            return vec4(pos,0); 
        }
        else  return iFrame < 10 ? vec4(1) : vec4(pix_stream(tex_coo,feedb,POS_SLIDE_AMT),0);
    }
    else if(tex_coo.y <= DRUMS_TOT_BLOCK_SIZE  && tex_coo.y >= DRUMS_BLOCK_SIZE)
    {
        int id =tex_coo.y - DRUMS_BLOCK_SIZE;
        ivec2 combo = DRUMS_COMBINATIONS[id];
        //first pix is most recent pos, rope attachments are read from the most recent pix 
        //at both ends and interpolated based on the pix pos on the row ( pix x pos) 
        ivec2 tcoo1 = ivec2(tex_coo.x, combo.x), tcoo2 = ivec2((ROPE_POINTS-1) - tex_coo.x, combo.y);
        //if I don't add this : ivec2(max(tcoo2.x,0),tcoo2.y) the texeleFetch won't work, not sure why
        vec3 pos1 = texelFetch(feedb,tcoo1,0).xyz, pos2 = texelFetch(feedb,ivec2(max(tcoo2.x,0),tcoo2.y),0).xyz;
        float inter = float(tex_coo.x)/(fROPE_POINTS-1.);
        vec3 pos = mix(pos1,pos2,inter);
        //fake gravity
        float inter_tri = tri(inter); 
        pos.y -= pow(inter_tri,0.5)*0.8;
        //inflate w/kick
        float kick = pow(texelFetch(feedb,ivec2(3,0),0).x,1.7);
        pos += mix(vec3(0),pos,inter_tri)*kick*1.;
        return iFrame < 10 ? vec4(1) : vec4(pos,0);
    }
}

void setVertexPosDrums(float amt, sampler2D feedb, sampler2D midi)
{
    const float rot_dist = 4.1;
    const vec3  accum_amt = vec3(0.4,0.5,0.6)*0.2;
    const vec3 accum_bias = vec3(0);
    for(int id = 0; id < NUM_DRUMS; id++)
    {
        vec3 acc = texelFetch(feedb,ivec2(0,id+1),0).xyz*accum_amt+accum_bias;
        vec3 fpos =DRUMS_FIX_POS[id]*1.;
        vec3 npos = normalize(fpos);
        
        vec3 nt = texelFetch(feedb,ivec2(id,0),0).xzw;
        Note note = Note(nt.x,0.,nt.y, nt.z);
        vec3 pos = fpos+cross(npos,vec3(0,1,0))*rot_dist *note.env;
        pos += fpos*note.env*amt*1.;
        pos = erot(pos,npos,note.sector+note.vel);
        pos += acc;
        pos = z_appear(pos,DRUM_Z_DISPL,DRUMS_CHAN, midi);
        //pos.xz *= rotate(iTime*0.1);
        DRUMS_VERTEX_POS[id] = pos*get_CC_size(DRUMS_CHAN, midi);
    }
    
}

vec3 getDomRepeatDrums(vec3 p )
{
    //p.y = abs(p.y);
    //p.x = abs(p.x);
    p.z = abs(p.z);
    p.y -= 5.;
    //p.zy *= rotate(iTime*0.347*TAU*1.);
    //p.yz *= rotate(iTime*0.347*TAU*2.);
    return p;
}

float z_plane_repeat(float z, float shift)
{
    bool left =  z > 0.;
    z += left ?  -shift : shift;
    return abs(z);
}
vec3 getDomRepeatMid(vec3 p, sampler2D txt, float offset, inout vec3 op, inout float highlight, inout vec2 uv)
{   
    // float curvature =0.;//pow(clamp(length(op.xy),0.,fPOS_HISTORY)/fPOS_HISTORY,1.5)*10.4;
    // bool left = op.z > 0.;
    // op.z += left ? -curvature : curvature;
    op.z = z_plane_repeat(op.z, offset);
    uv = op.xy/fPOS_HISTORY;
    uv = uv+0.5;
    uv *= 0.5; 
    uv.y += 0.25;
    
    op.xy = fract(op.xy*1.)+0.5;
    float sid = clamp(length(p.xy),0.,fPOS_HISTORY);
    vec3 pos1 = texelFetch(txt,ivec2(sid   ,MID_ENV_ROW+1),0).xyz,
         pos2 = texelFetch(txt,ivec2(sid+1.,MID_ENV_ROW+1),0).xyz,
         pos = mix(pos1,pos2,fract(sid));
    highlight = pos.z*3.;//smoothstep(0.4,0.9,pow(pos.z*1.,3.5));
    
    return pos;
}

vec4 lenses(vec3 p, sampler2D txt, float offset, inout bool is_outside)
{
    vec3 op = p;
    float highlight = 0.; 
    vec2 puv;
    vec3 pos = getDomRepeatMid(p,txt, offset, op, highlight, puv);
    float pl = op.z-pos.z;
    //carve out sphere: you might need to adjust radius and offset...
    float carve_out = length(vec3(p.xy,abs(p.z))+vec3(0,0,14.5))-50.;
    is_outside = pl > carve_out && abs(p.z) > offset;
    pl = max(pl,carve_out);
    return vec4(pl,puv,highlight);

}

vec3 getPosAnimMid(int id, float amt, sampler2D feedb, sampler2D midi)
{
    float env = texelFetch(feedb,ivec2(0,MID_ENV_ROW),0).x*1.;
    float env_foll = texelFetch(feedb,ivec2(0,MID_ENV_ROW+1),0).w*1.+0.;
    //float b = getCCval(10,MID_CHAN,midi);
    //b = bounce(b,30.)*2.-1.;
    //b = mix(pow(b,2.),pow(b,0.5),0.5)*2.-1.;
    //b = ease(b)*2.-1.;
    vec3 pos = vec3(0,0,-env_foll*1.);//vec3(b)*1.*(env_foll*10.);//+vec3(env*amt)+vec3(0,1,0); 

    return pos;
}

vec4 animDataMid(ivec2 tex_coo, sampler2D midi, sampler2D feedb)
{
    //if(tex_coo.x > 0) return vec4(0);
    if(tex_coo.y == MID_ENV_ROW)
    {
        int id = tex_coo.x;
        int chan = 2;
        int row = tex_coo.y ;
        float dur = 4.;
        float env_foll1 =  getCCval(12,MID_CHAN,midi)*0.5;
        float env_foll2 =  getCCval(13,MID_CHAN,midi);
        float env_foll3 =  getCCval(11,MID_CHAN,midi);
        float env_foll =  env_foll1+env_foll2+env_foll3;
        return vec4(getEnvelope(id,chan, row,dur,ENV_SLIDE_AMT,midi,feedb),env_foll);
    }
    else if(tex_coo.y < DRUMS_TOT_BLOCK_SIZE+ MID_BLOCK_SIZE)
    {
        if (tex_coo.x == 0)
        {
            int id = tex_coo.y-1;
            float anim_amt = 1.9;
            vec3 cur  = texelFetch(feedb,tex_coo,0).xyz;
            vec3 tar = getPosAnimMid(id,anim_amt,feedb, midi);
            vec3 pos = slide(cur,tar,0.6);
            float prev_env = texelFetch(feedb,tex_coo-ivec2(0,1),0).w;
            float cur_env = getCCval(12,2,midi);
            float tar_acc = cur_env - prev_env;
            float cur_acc = texelFetch(feedb,tex_coo,0).w;
            float acc = slide(cur_acc,tar_acc,0.4);
            return vec4(pos,acc); 
        }
        else  return iFrame < 10 ? vec4(1) : vec4(pix_stream(tex_coo,feedb,POS_SLIDE_AMT),0);
    }
    
}

vec3 getDomRepeatMidHi(vec3 p, int id, sampler2D txt, inout vec3 op, inout float sid)
{
    op.z = z_plane_repeat(op.z,0.1);
    float ang = 1./fPOS_HISTORY;
    float fa = (atan(p.y,p.x)/TAU+0.5)/ang  ;
    float sect = floor(fa);
    op.xy *= rotate(sect*ang*TAU+ang*0.5*TAU);
    sid = sect;//clamp(p.y,-50.,50.)/50.*fPOS_HISTORY;
    vec3 pos1 = texelFetch(txt,ivec2(sid,MID_HI_ENV_ROW+id+1),0).xyz,
         pos2 = texelFetch(txt,ivec2(sid+1,MID_HI_ENV_ROW+id+1),0).xyz,
         pos = mix(pos1,pos2,fract(sid));

    return pos;
}

vec3 getPosAnimMidHi(int id,  float env, sampler2D feedb, sampler2D midi)
{
    float cc = getCCval(10+id,MID_HI_CHAN, midi);
    float cc2 = getCCval(14,MID_HI_CHAN, midi);
    float ang = cc*TAU;
    env = pow(env,0.5);

    vec3 pos = vec3(abs(cos(ang)), 0, sin(ang))*1.5
              +vec3(float(id)/3.*1.-2., 0, float(id)/3.*(2.*cc2))*9.; 
    //pos = vec3(-1,0,0);
    //pos.x += env*100.1;
    //pos.yz *= rotate(iTime*5.);
    pos = z_appear(pos, MID_HI_Z_DISPL, MID_HI_CHAN,midi);
    return pos;
}
vec4 animDataMidHi(ivec2 tex_coo, sampler2D midi, sampler2D feedb)
{
    //if(tex_coo.y == MID_HI_ENV_ROW) return vec4(1);
    int block_begin = DRUMS_TOT_BLOCK_SIZE+ MID_TOT_BLOCK_SIZE;
    int block_end = block_begin  + MID_HI_TOT_BLOCK_SIZE; 
    if(tex_coo.y == MID_HI_ENV_ROW)
    {
        int id = tex_coo.x;
        int chan = MID_HI_CHAN;
        int row = tex_coo.y ;
        float dur = 2.;
        //float cur = texelFetch(feedb,tex_coo,0).w;
        //float env_foll = slide(cur,tar,0.9);
        return vec4(getEnvelopeGivenPitch(id ,chan, row,dur,ENV_SLIDE_AMT,midi,feedb),0);   
    }
    else if(tex_coo.y < block_end)
    {
        if (tex_coo.x == 0)
        {
            int id = tex_coo.y - block_begin - 1;
            float tar_env = texelFetch(feedb,ivec2(id,MID_HI_ENV_ROW),0).x;
            float cur_env = texelFetch(feedb,ivec2(id,MID_HI_ENV_ROW),0).w;
            float env = slide(cur_env,tar_env,0.5);
            vec3 tar = getPosAnimMidHi(id, env, feedb, midi);
            vec3 cur = texelFetch(feedb,tex_coo,0).xyz;
            vec3 pos = slide(cur, tar,1.);
            return vec4(pos,tar_env);
        }
        else  return iFrame < 10 ? vec4(1) : vec4(pix_stream(tex_coo,feedb,POS_SLIDE_AMT)+vec3(0,0.01,0),0);
    }
    
}

vec3 getDomRepeatBass(vec3 p)
{
    //p.x -=8.;
    //p.y = abs(p.y);
    //p.xz *= rotate(0.5);
    //p.z = abs(p.z);
    //p.y -=0.9;
    //p.yz *= rotate(iTime*1.18);
    return p;
}

void setVertexPosBass(float amt, sampler2D feedb, sampler2D midi)
{
    const float rot_dist = 0.81;
    float x = pow(getCCval(11,BASS_CHAN,midi),2.5)*PI*0.5, 
          y = pow(getCCval(10,BASS_CHAN,midi),0.5)*PI*0.5, 
          z = pow(getCCval(12,BASS_CHAN,midi),0.5)*PI;
    for(int id = 0; id < NUM_BASS; id++)
    {
        vec3 fpos =BASS_FIX_POS[id]*1.;
        vec3 npos = normalize(fpos);
        vec3 pos = fpos+cross(npos,vec3(0,1,0))*rot_dist ;
        vec2 nt = texelFetch(feedb,ivec2(id,BASS_ENV_ROW),0).xz;
        Note note = Note(nt.x,0.,nt.y, 0.);
        pos += fpos*note.env*amt*10.;
        pos = erot(pos,npos,note.vel);
        pos.xz *= rotate(x),pos.xy *= rotate(y),pos.yz *= rotate(z);
        //add z displacement for bass to appear/disappear based on song section
        pos = z_appear(pos, BASS_Z_DISPL, BASS_CHAN, midi);
        //DEBUG
        BASS_VERTEX_POS[id] = pos*get_CC_size(BASS_CHAN,midi);
    }
    
}

vec3 getPosAnimBass(int id, float amt, sampler2D feedb)
{
    vec3 pos = BASS_VERTEX_POS[id];
    for(int i = 0; i < NUM_BASS; i++)
    {
        if(id == i) continue;
        float env = texelFetch(feedb,ivec2(i,BASS_ENV_ROW),0).x;
        //DEBUG
        pos += texelFetch(feedb,ivec2(0,BASS_BLOCK_OFFSET+i),0).xyz*0.8*env; 
          
    }
    return pos;
}

vec3 getBassObjDispl(sampler2D midi)
{
    vec3 pos;
    bool first_bass = getCCval(15,BASS_CHAN,midi) < 0.5;
    if(first_bass)
    {
        float x = pow(getCCval(11,BASS_CHAN,midi),2.5), 
              y = pow(getCCval(10,BASS_CHAN,midi),0.5), 
              z = pow(getCCval(12,BASS_CHAN,midi),0.5);
        pos = vec3(x-0.5,z+y*0.5,y)*10.;
    }
    else
    {
        float x = pow(getCCval(11,BASS_CHAN,midi),0.5), 
              y = pow(getCCval(10,BASS_CHAN,midi),1.0), 
              z = pow(getCCval(12,BASS_CHAN,midi),2.0),
              a = 1.-getCCval(13,BASS_CHAN,midi),
              b = pow(getCCval(14,BASS_CHAN,midi),0.4);
        pos = vec3(x,z*0.3,y+a*0.2);
        pos.y += b;
        pos = pos*4.-2;

    }
    return vec3(0);//temp DEBUG 
}
vec4 animDataBass(ivec2 tex_coo, sampler2D midi, sampler2D feedb)
{
    //return vec4(float(tex_coo.y - BASS_BLOCK_OFFSET)/4.,0,0,0);
    if(tex_coo.y == BASS_ENV_ROW)
    {
        int id = tex_coo.x;
        int chan = BASS_CHAN;
        int row = BASS_ENV_ROW;
        float dur = 2.;
        float width = 4.;
        return vec4(getEnvelopeGivenPitch(id ,chan, row,dur,ENV_SLIDE_AMT,midi,feedb),0);
    }
    else if(tex_coo.y < BASS_BLOCK_OFFSET + BASS_BLOCK_SIZE )
    {
        //return vec4(1,0,0,0);
        if (tex_coo.x == 0)
        {
            int id = tex_coo.y-BASS_BLOCK_OFFSET-1;
            float anim_amt = 0.2;
            vec3 pos = getPosAnimBass(id,anim_amt,feedb);

            return vec4(pos,0); 
        }
        else  return iFrame < 10 ? vec4(1) : vec4(pix_stream(tex_coo,feedb,POS_SLIDE_AMT),0);
    }
    else if(tex_coo.y < BASS_BLOCK_OFFSET + BASS_TOT_BLOCK_SIZE+1  && tex_coo.y >= BASS_BLOCK_OFFSET + BASS_BLOCK_SIZE)
    {
        //return vec4(0.5);
        //return texelFetch(feedb,tcoo1,0).xyzz;
        int id =tex_coo.y - (BASS_BLOCK_OFFSET + BASS_BLOCK_SIZE);
        //return vec4(float(id) / 4. , 0 ,0,0);
        ivec2 combo = BASS_COMBINATIONS[id]+ivec2(BASS_BLOCK_OFFSET);
        //first pix is most recent pos, rope attachments are read from the most recent pix 
        //at both ends and interpolated based on the pix pos on the row ( pix x pos) 
        ivec2 tcoo1 = ivec2(tex_coo.x, combo.x), tcoo2 = ivec2((ROPE_POINTS-1) - tex_coo.x, combo.y);
        //if I don't add this : ivec2(max(tcoo2.x,0),tcoo2.y) the texeleFetch won't work, not sure why
        vec3 pos1 = texelFetch(feedb,tcoo1,0).xyz, pos2 = texelFetch(feedb,ivec2(max(tcoo2.x,0),tcoo2.y),0).xyz;
        float inter = float(tex_coo.x)/(fROPE_POINTS-1.);
        vec3 pos = mix(pos1,pos2,inter);
        //fake gravity
        float inter_tri = tri(inter); 
        //pos.y += pow(inter_tri,0.5)*0.8;
        return iFrame < 10 ? vec4(1) : vec4(pos,0);
    }
}
struct HitInfo
{
    float dist;
    //id is element id  and sub element id (e.g. rope id)
    ivec2 id;
    //returns rope/sphere connection interpolation index
    float smin;
    vec3 world_pos;
    vec3 nor;
    vec2 uv;
    vec2 uv_transorm;
    vec3 col;
    float env;
    float highlight;
};



/*
=============================================================================================================
LIGHT
=============================================================================================================
*/

float fresnel(float bias, float scale, float power, vec3 I, vec3 N)
{
    return bias + scale * pow(1.0 + dot(I, N), power);
}

//https://www.shadertoy.com/view/7lsBR8
// Very lame sky with sun-like object. Basically, this was a quick hack to emulate
// the "Forest Blurred" cube map... It needs work. :)
vec3 getSky(vec3 rd, vec3 ld){

    float lgt = max(dot(rd, ld), 0.);
    vec3 sky = mix(vec3(.1, .05, .04), vec3(.95, .97, 1)*3., clamp((rd.y + .5),0., 1.))/1.5;
    sky = mix(sky, vec3(8), pow(lgt, 8.));
    return min(sky*vec3(1, 1.1, 1.3), 1.);
}
vec3 blinn_phong(vec3 p, vec3 rd, vec3 light, vec3 norm,  vec3 col_diffuse)
{
    vec3 col_ambient = vec3(0.8588, 0.8196, 0.098);
    vec3 col_specular = vec3(0.3686, 0.3725, 0.3059);
    return  col_ambient + 
            col_diffuse * max(dot(normalize(light-p),norm),0.)+ 
            col_specular * pow(max(dot(reflect(normalize(light-p),norm),rd),0.),2.);

}




