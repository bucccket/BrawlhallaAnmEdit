from DecodeAnm import AnmFile

if __name__ == "__main__":
    emote_anm = AnmFile("Animation_Emote.og.anm")
    character_select_anm = AnmFile("Animation_CharacterSelect.og.anm")
    
    a__EmoteAnimation = emote_anm.anm_classes["Animation_Emote.swf/a__EmoteAnimation"]
    a__CharacterSelectAnimation = character_select_anm.anm_classes["Animation_CharacterSelect.swf/a__CharacterSelectAnimation"]
    
    name = a__EmoteAnimation.animations[33].name
    a__EmoteAnimation.animations[33] = a__CharacterSelectAnimation.animations[98]
    a__EmoteAnimation.animations[33].name = name
    
    emote_anm.Save("Animation_Emote.anm")