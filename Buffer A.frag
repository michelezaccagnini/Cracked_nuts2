//MIDI data parsing

#define FEEDBACK iChannel0
#define MIDI iChannel1

//debug: visualize midi texture
#if 0
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragColor = texelFetch(MIDI,ivec2(fragCoord),0);
}
#else

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    ivec2 iCoo = ivec2(fragCoord);
    ivec2 blocks_dim = ivec2(POS_HISTORY ,BASS_BLOCK_OFFSET + BASS_TOT_BLOCK_SIZE);                            
    if(any(greaterThanEqual(iCoo,blocks_dim))) discard;
    float amt = 0.2;
    setVertexPosDrums(amt,FEEDBACK, MIDI);
    setVertexPosBass (amt,FEEDBACK, MIDI);
    fragColor =     iCoo.y < MID_BLOCK_OFFSET ? 
                    animDataDrums(iCoo,MIDI,FEEDBACK) : 
                    iCoo.y < MID_HI_BLOCK_OFFSET ?  
                    animDataMid(iCoo,MIDI,FEEDBACK) :
                    iCoo.y < BASS_BLOCK_OFFSET? 
                    animDataMidHi(iCoo,MIDI,FEEDBACK) : animDataBass (iCoo,MIDI,FEEDBACK);

}
#endif