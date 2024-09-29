namespace Drumwise.API;

internal static class ApiPaths
{
    internal const string MainPath = "api/v2/";

    #region Homework

    internal const string HomeworkRootApi = "homework";

    internal const string CreateHomework = "/";
    internal const string GetAllHomeworks = "/";
    internal const string GetHomeworkById = "/{{HomeworkId}}";

    #endregion

    #region Identity

    internal const string ManageAccountRootApi = "manage";
    
    internal const string CustomRegister = "customRegister";
    internal const string AddAdditionalUserData = "/addAdditionalUserData";

    #endregion

    #region MidiConverter

    internal const string MidiConverterRootApi = "midiconverter";
    
    internal const string InitiateAudioConverting = "/initiateAudioConverting";

    #endregion
}