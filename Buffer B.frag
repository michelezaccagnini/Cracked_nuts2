#define BKGR iChannel0
#define MIDI iChannel1
#define BUF_A iChannel2
#define BUF_C iChannel3



HitInfo map(vec3 p)
{
    HitInfo res;
    res.id  = ivec2(-1);
    res.dist = MAX_DIST;
    #ifdef BACK_PLANE
    res.dist = min(res.dist,sdPlane(p-POS_PLANE_Y));
    res.dist = min(res.dist,sdPlaneZ(p-POS_PLANE_Z));
    #endif
    //mid plane
    {
        // vec3 op = p;
        // float highlight = 0.; 
        // vec2 puv;
        // vec3 pos = getDomRepeatMid(p,BUF_A, MID_Z_DISPL, op, highlight, puv);
        // float pl = op.z-pos.z;
        // float carve_out = length(vec3(p.xy,abs(p.z))+vec3(0,0,14.5))-50.;
        // bool outside = pl > carve_out && abs(p.z) > 25.;
        // pl = max(pl,carve_out);
        bool is_outside = false;
        vec4 lens = lenses(p,BUF_A,MID_Z_DISPL,is_outside);
        bool this_plane = lens.x < res.dist;
        if(this_plane)
        {
            res.dist = min(res.dist, lens.x), 
            res.id = ivec2(0,3), 
            res.uv = lens.yz,
            res.highlight = lens.w;
        }
        if(is_outside) return res;
    }  
    //drums
    {
        vec3 op = getDomRepeatDrums(p);
        //drum ropes
        for(int id = 0; id < NUM_DRUMS_COMBINATIONS; id++)
        {
            //env on first row of buf A, horizontally
            //but drums combo is 6 while num drums is 4...?
            float env = texelFetch(BUF_A,ivec2(id,0),0).x;
            int i = id + DRUMS_BLOCK_SIZE;
            vec3 hit_point = vec3(0);
            float thick = DRUM_THICK;
            // rope pos are on rows from 9 to 37 iof BUF A
            vec3 rop = rope(op,i,BUF_A,env, thick,hit_point);
            
            bool this_rope = rop.x < res.dist;
            if(this_rope)
            {
                float kick = texelFetch(BUF_A,ivec2(3,DRUMS_ENV_ROW),0).x;
                res.id = ivec2(id,1), res.uv = rop.yz, 
                //glow += max(glo*env,vec3(0)),
                res.env = env, 
                res.dist = rop.x;
                
            }
        }
        //drum spheres
        for(int y = 0; y < NUM_DRUMS; y++)
        {
            //offset texelfetch y coordionate by 1 since first row is envelopes
            vec3 pos = texelFetch(BUF_A,ivec2(0,y+1),0).xyz;
            float sph = length(op-pos)-0.01;
            float h =0.;
            bool this_sph = sph < res.dist;
            res.dist = smin(res.dist,sph,sph_smin, h);
            if(this_sph) 
            {
                    res.smin = h, res.id = ivec2(y,2), res.smin = h;
            } 
        }
    }
    //mid-hi objects
    {
        for(int id = 0; id < NUM_MID_HI; id++)
        {
            vec3 op = p;
            //p.xz *= rotate(iTime*0.2);
            float  sect_id = 0.;
            float env = texelFetch(BUF_A, ivec2(id, MID_HI_ENV_ROW),0).x;
            env = pow(env,0.8);
            vec3 pos = getDomRepeatMidHi(p, id, BUF_A,op,sect_id);
            float size = abs(env*fPOS_HISTORY-sect_id)/fPOS_HISTORY;
            size  = size+0.4;
            float sph = length(op-pos)-MID_HI_RAD*size;
            bool this_sph = sph < res.dist;
            if(this_sph)
            {
                res.dist = sph, 
                res.env = env,
                res.world_pos = pos,
                res.id = ivec2(sect_id,4);
            }
        }
    }  
    //bass
    {
        vec3 op = getDomRepeatBass(p);
        //op.x = abs(op.x);
        //op -= getBassObjDispl(MIDI);
        for(int id = 0; id < NUM_BASS_COMBINATIONS; id++)
        {
            //env on first row of buf A, horizontally
            int idm = int(mod(float(id),float(NUM_BASS)));
            float env = texelFetch(BUF_A,ivec2(idm,BASS_ENV_ROW),0).x;
            int i = id + BASS_BLOCK_OFFSET + BASS_BLOCK_SIZE;
            vec3 hit_point = vec3(0);
            float thick = BASS_THICK;
            // rope pos are on rows from 9 to 37 iof BUF A
            vec3 rop = rope(op,i,BUF_A,env,thick,hit_point);
            
            bool this_rope = rop.x < res.dist;
            if(this_rope)
            {
                res.id = ivec2(id,5), res.uv = rop.yz, 
                res.world_pos = hit_point, 
                res.env = env, 
                res.dist = rop.x;
            }
        }
    }
    //ray marching correction for rope
    res.dist *= 0.7;
    return res;
}

float mapSimple(vec3 p)
{
   
    float dist = MAX_DIST;
    #ifdef BACK_PLANE
    dist = min(dist,sdPlane(p-POS_PLANE_Y));
    dist = min(dist,sdPlaneZ(p-POS_PLANE_Z));
    #endif
    //mid plane
    {
        bool is_outside = false;
        vec4 lens = lenses(p,BUF_A,MID_Z_DISPL,is_outside);
        bool this_plane = lens.x < dist;
        dist = min(dist,lens.x);
        if(is_outside) return dist;
    }
    //drums
    {
        vec3 op = getDomRepeatDrums(p);
        for(int id = 0; id < NUM_DRUMS_COMBINATIONS; id++)
        {
            //env on first row of buf A, horizontally
            float env = texelFetch(BUF_A,ivec2(id,0),0).x;
            int i = id + DRUMS_BLOCK_SIZE;
            vec3 hit_point = vec3(0);
            float thick = DRUM_THICK;
            // rope pos are on rows from 9 to 37 iof BUF A
            vec3 rop = rope(op,i,BUF_A,env,thick,hit_point);
             dist = min(rop.x,dist);
        }
        
        for(int y = 0; y < NUM_DRUMS; y++)
        {
            //offset texelfetch y coordionate by 1 since first row is envelopes
        vec3 pos = texelFetch(BUF_A,ivec2(0,y+1),0).xyz;
        float sph = length(op-pos)-0.01;
        float h =0.;
        dist = smin(dist,sph,sph_smin, h);
        }
    }

    //mid-hi objects
    { 
        for(int id = 0; id < NUM_MID_HI; id++)
        {
            vec3 op = p;
            float sect_id = 0.;
            vec3 pos = getDomRepeatMidHi(p, id, BUF_A,op, sect_id);
            float env = texelFetch(BUF_A, ivec2(id, MID_HI_ENV_ROW),0).x;
            env = pow(env,0.8);
            float size = abs(env*fPOS_HISTORY-sect_id)/fPOS_HISTORY;
            size  = size+0.4;
            float sph = length(op-pos)-MID_HI_RAD*size*env*5.;
            bool this_sph = sph < dist;
            if(this_sph)
            {
                dist = sph;
            }
        }
    }   
    //bass
    {
        vec3 op = getDomRepeatBass(p);
        //op.x = abs(op.x);
        // /op -= getBassObjDispl(MIDI);
        for(int id = 0; id < NUM_BASS_COMBINATIONS; id++)
        {
            //env on first row of buf A, horizontally
            float env = texelFetch(BUF_A,ivec2(id,BASS_ENV_ROW),0).x;
            int i = id + BASS_BLOCK_OFFSET + BASS_BLOCK_SIZE;
            vec3 hit_point = vec3(0);
            float thick = BASS_THICK;
            // rope pos are on rows from 9 to 37 iof BUF A
            vec3 rop = rope(op,i,BUF_A,env,thick,hit_point);
            
            bool this_rope = rop.x < dist;
            if(this_rope)
            {
                dist = rop.x;
            }
        }
    }
    return dist;
}

HitInfo intersect(vec3 ro, vec3 rd)
{
    HitInfo res;
    float d = 0.002;
    for(int i = 0; i < 80; i++)
    {
        vec3 p = ro +rd*d;
        res = map(p);
        
        if(d > MAX_DIST || abs(res.dist) < 0.01) break;
        d += res.dist;
    }
    res.dist =d;
    return res;
}

vec3 normal(vec3 p) {
    vec2 e = vec2(1e-2, 0.);
    //vec2 e = vec2(0.14, 0.);

    vec3 n = mapSimple(p) - vec3(
        mapSimple(p-e.xyy),
        mapSimple(p-e.yxy),
        mapSimple(p-e.yyx)
    );

    return normalize(n);
}

vec3 getLight(vec3 p, vec3 ro, vec3 rd, vec3 n, HitInfo hit, float night, bool is_refl)
{
    vec3 col = texture(BKGR,n).xyz*0.51*night;
    vec3 light1Dir = normalize(vec3(0.0,0, 0.2));
    vec3 light2Dir = normalize(vec3(0.0,0, -0.2));
    vec3 light1Color = normalize(vec3(3, 2, 1))*0.8;
    vec3 light2Color = normalize(vec3(1, 2, 3))*0.8;
    col += pow(clamp(dot(reflect(rd, n), light1Dir),0.,1.), 0.8) * light1Color * 1.*night ;
    col += pow(clamp(dot(reflect(rd, n), light2Dir),0.,1.), 0.8) * light1Color * 1. *night;
    if(hit.id.y == 3)
    {
        col =texture(BUF_C, hit.uv).xyz*10.;
        if(!is_refl)
        {
            col *= 0.5;
            col += pow(hit.highlight,2.5)*1.0*vec3(0.5,0.2,0.5);
        }else{col *= .5;}
        //col = hit.uv.xxx;
    }
    if(hit.id.y == 4)
    {
        float sid = float(hit.id.x);
        float dd= length(hit.world_pos - ro);
        vec3 glow =  vec3(0.15,0.5,0.2)*5.;
        float ill = abs(pow(hit.env,0.2)*fPOS_HISTORY-sid)/fPOS_HISTORY;
        float mult = smoothstep(0.05,0., ill);
        col += glow*mult*10.;
    }
    if(hit.id.y == 5)
    {
        float ill = smoothstep(0.051,.01,abs(pow(hit.env,0.6)-(0.4-hit.uv.y*0.4)));
        vec3 ilc = normalize(vec3(1,2,3));
        col += ill*ilc;
    }
    return max(col,vec3(0));
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    ivec2 iCoo = ivec2(fragCoord);
    vec2 uv =2.*(fragCoord-iResolution.xy*0.5)/iResolution.y;
    float kick = pow(texelFetch(BUF_A,ivec2(3,DRUMS_ENV_ROW),0).x,2.5);
    vec3 ro = false ? vec3(1,20,0) : vec3(cos(iTime*0.2),sin(iTime*0.2),0)*30.;
    ro = getRO(MIDI);
    //turn of lights
    float night = getCCval(15,RAY_OR_CHAN, MIDI);
    vec3 lookat = getLookat(iTime, kick,MIDI);
    mat3 cam = camera(ro, lookat,0.0);
    vec3 rd = cam*vec3(uv,1.0);
    HitInfo hit = intersect(ro,rd);
    vec3 col = texture(BKGR,rd).xyz*0.5*night;
    if(hit.dist < MAX_DIST)
    {
        vec3 p = ro + rd*hit.dist;
        vec3 n = normal(p);
        bool is_refl = false;
        col = max(getLight(p, ro,rd,n, hit, night, is_refl),vec3(0));
        if(hit.id.y == 3)
        {
            col *= 0.2;
            is_refl = true;
            vec3 rrd = reflect(rd,n);
            vec3 rro = p + rrd*0.1;
            col += texture(BKGR,rrd).xyz*0.5;
            HitInfo ref = intersect(rro,rrd);
            vec3 pp = rro + rrd*ref.dist;
            vec3 rn = normal(pp);
            col += max(getLight(pp, rro, rrd,rn, ref,night, is_refl),vec3(0));
            vec3 txt = texture(BUF_C,abs(p.xz*0.1)).xyz;
            //col = mix(col, txt, 0.2);
            //col = abs(p.xxx*0.01);
        }
        
    } 
    #if 0
    for(int i = 0; i < NUM_DRUMS; i++)
    {
        vec3 lig = texelFetch(BUF_A,ivec2(0,i+1),0).xyz;
        float lig_env = texelFetch(BUF_A,ivec2(i,0),0).x;
        vec3 flash = max(integrateLightFullView(ro-lig,rd,0.9,0.55),0.)*vec3(0.8)*pow(lig_env,1.2);
        //col *= flash+0.1;
        col +=flash;
    }
    #endif
    //col = pow(col,vec3(0.8545));
    //vec3 feed = max(texture(FEEDBACK,fragCoord/iResolution.xy).xyz,vec3(0));
    //col = mix(feed,col,0.8-(1.*kick));
    //col = vec3(0,cos(t1),cos(t2))*(0.004+kick*0.5);
    fragColor = vec4(col,0);

}

