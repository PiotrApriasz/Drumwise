namespace Drumwise.Application.Common.Models.Identity;

public record AdditionalUserDataRequest(string Name, string Surname, string Role, float? Experience);