namespace Drumwise.API;

internal static class ApiPaths
{
    internal const string MainPath = "api/v2/";

    #region Homework

    internal const string HomeworkRootApi = "homework";

    internal const string CreateHomework = HomeworkRootApi;
    internal const string GetAllHomeworks = HomeworkRootApi;
    internal const string GetHomeworkById = $"{HomeworkRootApi}/{{HomeworkId}}";

    #endregion

    #region Identity

    internal const string ManageAccountRootApi = "manage";
    
    internal const string CustomRegister = "customRegister";
    internal const string AddAdditionalUserData = "addAdditionalUserData";

    #endregion
}